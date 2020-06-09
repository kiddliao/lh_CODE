__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
import pycocotools.mask as maskUtils
import copy
from scipy import interpolate

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters 指定检测的参数(各个阈值,maxDets等等)
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds()) #写入标注框图片排序后的image_id
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats: #获取测试集有标注图片的标注
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag  部分比较小的物体,会设置忽略检测
        for gt in gts: 
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation 
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt) #key是image_id和category_id value是这张图片这个分类目标的annotations
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results 积累结果

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        # self.ious = {(imgId, catId): computeIoU(imgId, catId) \
        #                 for imgId in p.imgIds
        #                 for catId in catIds}  #双层循环,对每张图片检测每个类
        self.lh_ious = {(imgId, catId): self.lh_computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds} #双层循环,对每张图片检测每个类
        isum = [0]*len(catIds)
        
                    
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        # self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
        #          for catId in catIds
        #          for areaRng in p.areaRng #全部尺寸 小中大 一共四个尺寸 [0,1024,9216,1e10]
        #          for imgId in p.imgIds
        #      ]
        self.evalImgs = [self.lh_evaluateImg(imgId, catId, areaRng, maxDet)
        # self.lh_evalImgs = [self.lh_evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng 
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0: #如果这张图片的这个类别的目标 gt框和检测框都没有就直接返回
            return []
        if len(gt) and len(dt) and imgId:
            pass
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort') #检测框标注的score就是置信度,按置信度排序,加负号是为了从大到小排序
        dt = [dt[i] for i in inds] #排序
        if len(dt) > p.maxDets[-1]: #如果检测框数量大于最大检测数 选置信度大的前maxDets[-1]也就是100个
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt] #iscrowd为0和1是两种分割标注的格式,coco的目标检测数据集的标注iscrowd一律为0
        ious = maskUtils.iou(d,g,iscrowd) #计算这张图这个分类的预测框和gt框的iou ious格式为(len(dt),len(gt)) 也就是这个预测框和所有gt框的iou
        return ious

    def lh_computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0: #如果这张图片的这个类别的目标 gt框和检测框都没有就直接返回
            return []
        if len(gt) and len(dt) and imgId:
            pass
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort') #检测框标注的score就是置信度,按置信度排序,加负号是为了从大到小排序
        dt = [dt[i] for i in inds] #排序
        # if len(dt) > p.maxDets[-1]: #如果检测框数量大于最大检测数 选置信度大的前maxDets[-1]也就是100个
        #     dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt] #iscrowd为0和1是两种分割标注的格式,coco的目标检测数据集的标注iscrowd一律为0
        ious = maskUtils.iou(d,g,iscrowd) #计算这张图这个分类的预测框和gt框的iou ious格式为(len(dt),len(gt)) 也就是这个预测框和所有gt框的iou
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        # if len(gt) and len(dt) and gt[0]['image_id']==8:
        #     print()

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]): #目标尺寸在检测范围内才检测
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last. gtind前面都是ignore为0的gt 后面都是ignore为1的gt
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs) #iou阈值列表
        G = len(gt) #gt框
        D = len(dt) #dt框
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt]) #gt框是否被ignore
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs): #对于10个iou阈值的这张图片的这个分类的评估
                for dind, d in enumerate(dt): #取出一个预测框
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1 #如果m没重新赋值为1就跳出则unmatched即负样本
                    for gind, g in enumerate(gt): #取出一个gt框
                        # if this gt already matched, and not a crowd, continue 如果这个gt框已经和其他置信度高的检测框匹配则不再与其他检测框匹配
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop 如果这个检测框已经和某个gt框匹配则跳出循环
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made 如果iou小于阈值或者是小于当前匹配的最佳iou 意思是一个dt框要和gt框中iou最大的一个匹配
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately iou超过阈值或上一个匹配的iou的话更新
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt 匹配不成功继续循环 匹配成功保存变量
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m] #如果这个dt对应的最佳gt本身就是被ignore的，就把这个dt也设置为ignore
                    dtm[tind,dind]  = gt[m]['id'] #保存在第tind个iou阈值下和第dind个预测框匹配的gt框的annotations_id
                    gtm[tind,m]     = d['id'] #保存在第tind个iou阈值下和第m个gt框匹配的gt框的annotations_id
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def lh_evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        # if len(gt) and len(dt) and gt[0]['image_id']==8:
        #     print()

        for g in gt:
            g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last. gtind前面都是ignore为0的gt 后面都是ignore为1的gt
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.lh_ious[imgId, catId][:, gtind] if len(self.lh_ious[imgId, catId]) > 0 else self.lh_ious[imgId, catId]

        T = len(p.iouThrs) #iou阈值列表
        G = len(gt) #gt框
        D = len(dt) #dt框
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt]) #gt框是否被ignore
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs): #对于10个iou阈值的这张图片的这个分类的评估
                for dind, d in enumerate(dt): #取出一个预测框
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1 #如果m没重新赋值为1就跳出则unmatched即负样本
                    for gind, g in enumerate(gt): #取出一个gt框
                        # if this gt already matched, and not a crowd, continue 如果这个gt框已经和其他置信度高的检测框匹配则不再与其他检测框匹配
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop 如果这个检测框已经和某个gt框匹配则跳出循环
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made 如果iou小于阈值或者是小于当前匹配的最佳iou 意思是一个dt框要和gt框中iou最大的一个匹配
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately iou超过阈值或上一个匹配的iou的话更新
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt 匹配不成功继续循环 匹配成功保存变量
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m] #如果这个dt对应的最佳gt本身就是被ignore的，就把这个dt也设置为ignore
                    dtm[tind,dind]  = gt[m]['id'] #保存在第tind个iou阈值下和第dind个预测框匹配的gt框的annotations_id
                    gtm[tind,m]     = d['id'] #保存在第tind个iou阈值下和第m个gt框匹配的gt框的annotations_id
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs) #召回率阈值 AR的阈值为0-1 间隔0.01 ，一共101个阈值
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        xray_recall = np.zeros((T, K, 4, A, M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK] #将分类id从0开始
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds) #图片数
        A0 = len(_pe.areaRng) #面积阈值的数量
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list): #第k0个类
            Nk = k0*A0*I0 #之前已经过了Nk个图片和面积阈值
            for a, a0 in enumerate(a_list):
                Na = a0*I0 #之前已经过了Na面积阈值
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list] #evalImgs的长度是图片数*面积阈值数*分类数 取出当前分类当前尺寸下测试集所有图片的评估结果
                    E = [e for e in E if not e is None]  #去除空结果
                    lh_E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    lh_E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) #将结果连接起来
                    lh_dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) 
                    
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort') #置信度从大到小排序
                    dtScoresSorted = dtScores[inds]

                    lh_inds = np.argsort(-lh_dtScores, kind='mergesort') #置信度从大到小排序
                    lh_dtScoresSorted = lh_dtScores[inds]
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                   
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) ) #如果dtm对应的匹配gt不为0,且对应的gt没有被忽略,这个dt就是TP
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) ) #fp

                    lh_dtm  = np.concatenate([e['dtMatches'] for e in lh_E], axis=1)[:,inds] #voc评测指标不考虑maxDets
                    lh_dtIg = np.concatenate([e['dtIgnore'] for e in lh_E], axis=1)[:, inds]
                    lh_tps = np.logical_and(               dtm,  1 ) 
                    lh_fps = np.logical_and(np.logical_not(dtm), 1 ) 
                    
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    lh_tp_sum = np.cumsum(lh_tps, axis=1).astype(dtype=np.float)
                    lh_fp_sum = np.cumsum(lh_fps, axis=1).astype(dtype=np.float)

                    if a == 0 and m == 2 and k==4:
                        # print(k)
                        if np.count_nonzero(lh_dtm[0] == 0) != np.count_nonzero(dtm[0] == 0):
                            print(np.count_nonzero(lh_dtm[0] == 0), np.count_nonzero(dtm[0] == 0))
                        if np.count_nonzero(dtIg[0]):
                            print(np.count_nonzero(dtIg[0]))
                    for t, (tp, fp) in enumerate(zip(lh_tp_sum, lh_fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig #npig是这个分类gt框数量 rc是召回率 pr是精确率
                        pr = tp / (fp+tp+np.spacing(1))
                      
                        if nd not in (0,1):
                            xray_sens = tp / npig
                            xray_fp = fp / I0
                            xray_thresh = [0.05, 0.1, 0.2]
                            f = interpolate.interp1d(xray_fp, xray_sens, fill_value='extrapolate')
                            max_fpr = xray_fp[-1]
                            if xray_thresh[-1] < max_fpr:
                                th = len(xray_thresh)
                            else:
                                th=np.argwhere(np.array(xray_thresh)>=max_fpr)[0][0]
                            valid_avgFP=np.hstack((xray_thresh[:th],max_fpr))
                            xray_res = f(valid_avgFP)
                            xray_recall[t, k,:len(xray_res), a, m] = xray_res
                        else:
                            xray_recall[t, k,:, a, m] = [0, 0, 0, 0]

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig #npig是这个分类gt框数量 rc是召回率 pr是精确率
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))
                      
                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0
                            

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1): #voc2007的11插值法的改进101插值法
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left') #在求得的召回率中插入101个召回率阈值并且返回插入位置 即把召回率分块
                        try:
                            for ri, pi in enumerate(inds): 
                                q[ri] = pr[pi] #得到大于当前召回率阈值的最大精确率和对应的置信度
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'xray_recall':xray_recall
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<10} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            elif ap==0:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:, aind, mind]
            elif ap == 2:
                # tpr = self.eval['xray_recall']
                # xray_recall = tpr[0,:,:, 0, 2]
                # tmp = xray_recall.sum(axis=0) / 9
                # final_tmp = tmp[:3].mean()
                # lhfm = ' {:<18} {} @[ IoU={:<10} | area={:>6s} | maxDets={:>3s} ] = {:0.3f}'
                # print(lhfm.format('TPR[fpr=0.05]','ATPR','0.5','all','100',tmp[0]))
                # print(lhfm.format('TPR[fpr=0.10]','ATPR','0.5','all','100',tmp[1]))
                # print(lhfm.format('TPR[fpr=0.20]','ATPR','0.5','all','100',tmp[2]))
                # # print(lhfm.format('TPR[fpr=maxfpr]','ATPR','0.5','all','100',tmp[3]))
                # print(lhfm.format('Average TPR', 'ATPR', '0.5', 'all', '100', final_tmp))
                # return tmp[:3].tolist()+[final_tmp]
                pass

            if ap != 2: 
                if len(s[s>-1])==0:
                    mean_s = -1
                else:
                    mean_s = np.mean(s[s>-1]) #均值
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            tpr = _summarize(2)
            return stats,stats[1]
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats, mAP = summarize()
        return mAP

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None


if __name__ == '__main__':
    from pycocotools.coco import COCO
    # from lh_coco_eval import COCOeval
    VAL_GT = f'xray_pred2\\newinstances_val2017.json'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES] #设置最大检测图片数量

    def _eval(coco_gt, image_ids, pred_json_path):#image_ids是所有测试集有标注的图 pred_json_path是预测文件的路径
        # load results in COCO evaluation tool
        coco_pred = coco_gt.loadRes(pred_json_path)
        # run COCO evaluation
        print('所有类的BBox')
        coco_eval = COCOeval(coco_gt, coco_pred, 'bbox') #'bbox'是检测 'segm'是分割
        coco_eval.params.imgIds = image_ids #如果检测图片数大于MAX_IMAGES 修改当前gt图片的id在MAX_IMAGES之内
        coco_eval.evaluate() #评估
        coco_eval.accumulate()
        mAP = coco_eval.summarize()
        return [mAP]*4

        # for cat in coco_pred.cats.values():
        #     print("{cat['name']}类的BBOX")
        #     coco_eval.params.catIds = [cat['id']]
        #     coco_eval.params.imgIds = image_ids
        #     coco_eval.evaluate()
        #     coco_eval.accumulate()
        #     coco_eval.summarize()

    _eval(coco_gt, image_ids, 'xray_pred2\mocod_val1_88_55.json')