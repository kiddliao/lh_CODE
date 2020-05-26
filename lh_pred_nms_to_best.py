import json
import numpy as np
import os
import glob
from collections import defaultdict
import torch
from lh_coco_eval import COCOeval
from pycocotools.coco import COCO
import threading
import time
import copy

NAME='val'

def lh_IOU(box1, box2, x1y1x2y2=False, iou_type='IoU'):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # else:  # x, y, w, h = box1
    #     b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    #     b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    #     b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    #     b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:  # x1, y1, w, h = box1
            b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
            b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
            b2_x1, b2_x2 = box2[0], box2[0] + box2[2] 
            b2_y1, b2_y2 = box2[1], box2[1] + box2[3] 

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    iou = inter_area / union_area  # iou
    if iou_type in ['GIoU', 'CIoU', 'DIoU']:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if iou_type=='GIoU':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        if iou_type in ['CIoU','DIoU']:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if iou_type=='DIoU':
                return iou - rho2 / c2  # DIoU
            elif iou_type=='CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


def lh_NMS(prediction, conf_thresh=0.4, iou_thresh=0.5, style='OR',type='IoU'):
    '''
    prediction是预测结果
    confidence是objectness分数阈值,confidence大于此值才进行计算
    nms_conf是NMS的IoU阈值
    '''
    #coco是左上角坐标和宽高
    if torch.sum(prediction[:, 4] < conf_thresh) == prediction.size()[0]:
        return []
    
    conf_mask = (prediction[:, 4] >= conf_thresh)
    prediction = prediction[conf_mask,:]  #删除置信度小于阈值的预测框
    #宽高有问题可能会导致merge进入死循环
    w_true_ind = prediction[:, 2] >=1
    prediction = prediction[w_true_ind,:]
    h_true_ind = prediction[:, 3] >=1
    prediction = prediction[h_true_ind,:]
    

    #置信度排序
    indmax = torch.argsort(-1 * prediction[:, 4])
    prediction = prediction[indmax, :]

    if prediction.size()[0] == 1:
        return prediction.tolist()
    det_max = []

    # if style == 'OR':  #hard nms,直接删除相邻的同类别目标,密集目标的输出不友好
    #     for idx, bbox in enumerate(prediction):
    #         if prediction[idx][4] != 0:
    #             ious = lh_IOU(box_corner[idx,:4].unsqueeze(0), box_corner[idx + 1 :,:4])
                
    #             ind_ious_bool = (ious > iou_thresh)
    #             interplote = torch.zeros(idx + 1).bool()
    #             ind_ious_bool = torch.cat((interplote, ind_ious_bool))
    #             prediction[ind_ious_bool,:] *= 0
    #             box_corner[ind_ious_bool,:] *= 0
    dc = prediction.clone()
    if style == 'OR':
        # METHOD1
        # ind = list(range(len(dc)))
        # while len(ind):
        # j = ind[0]
        # det_max.append(dc[j:j + 1,:4].squeeze().tolist())  # save highest conf detection
        # reject = (lh_IOU(dc[j], dc[ind],iou_type=type) > iou_thresh).nonzero()
        # [ind.pop(i) for i in reversed(reject)]
        #METHOD2
        while dc.size()[0]:
            det_max.append(dc[:1].squeeze().tolist())
            if len(dc) == 1:
                break
            iou = lh_IOU(dc[0], dc[1:],iou_type=type)
            dc = dc[1:][iou <= iou_thresh]

    elif style == 'AND':  #and nms,在hard-nms的逻辑基础上,增加是否为单独框的限制,删除没有重叠框的框(减少误检)
        while dc.size()[0] > 1:
            iou = lh_IOU(dc[0], dc[1:],iou_type=type)
            if iou.max() > iou_thresh:
                det_max.append(dc[:1].squeeze().tolist())
            dc = dc[1:][iou <= iou_thresh]
    
    elif style == 'MERGE':  #merge nms,在hard-nms的基础上,增加保留框位置平滑策略(重叠框位置信息求解平均值),使框的位置更加精确
        while dc.size()[0]:
            if len(dc) == 1:
                det_max.append(dc.squeeze().tolist())
                break
            i = lh_IOU(dc[0], dc,iou_type=type) > iou_thresh
            weights = dc[i, 4:5]
            if torch.isnan(((weights * dc[i,:4]).sum(0) / weights.sum())[0]):
                print(1)
            dc[0,:4] = (weights * dc[i,:4]).sum(0) / weights.sum()
            det_max.append(dc[:1].squeeze().tolist())
            dc = dc[i == 0]
    
    elif style == 'SOFT': #soft nms,改变其相邻同类别目标置信度,后期通过置信度阈值进行过滤,适用于目标密集的场景
        sigma = 0.5
        while dc.size()[0]:
            if len(dc) == 1:
                det_max.append(dc.squeeze().tolist())
                break
            det_max.append(dc[:1].squeeze().tolist())
            iou = lh_IOU(dc[0], dc[1:],iou_type=type)
            dc = dc[1:]
            dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  #decay confidences
            dc = dc[dc[:, 4] > iou_thresh]  # new line per https://github.com/ultralytics/yolov3/issues/362
    return det_max

def count_num(ensembled_pred,j):
    nums=0
    for i in ensembled_pred.keys():
        nums += len(ensembled_pred[i][j])
    print(nums)
    return nums
           
def lh_defaultlist():
    return defaultdict(list)

def trans_2_standard_ann(res ,flag = 0):
    sample={}
    all_labels = []
    nums = 0
    for k, v in res.items():
        for a, b in v.items():
            if len(b) != 0:
                nums += len(b)
                for i in range(len(b)):
                    sample['bbox'] = res[k][a][i][:4]
                    sample['category_id'] = a
                    sample['image_id'] = k
                    sample['score'] = res[k][a][i][4]
                    all_labels.append(sample.copy())                
    if flag == 1:
        with open('new_{}.json'.format(NAME), 'w') as f:
            json.dump(all_labels, f)
    return all_labels, nums

def Copy(old):
    # new = defaultdict(lh_defaultlist)
    # for k, v in old.items():
    #     for a, b in v.items():
    #         new[k][a] = b.copy()
    # return new
    return copy.deepcopy(old)

def _eval(coco_gt, image_ids, pred_json_path):#image_ids是所有测试集有标注的图 pred_json_path是预测文件的路径
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)
    # run COCO evaluation
    print('所有类的BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox') #'bbox'是检测 'segm'是分割
    coco_eval.params.imgIds = image_ids #如果检测图片数大于MAX_IMAGES 修改当前gt图片的id在MAX_IMAGES之内
    coco_eval.evaluate() #评估
    coco_eval.accumulate()
    tpr = coco_eval.summarize()
    
    for cat in coco_pred.cats.values():
        print(f"{cat['name']}类的BBOX")
        coco_eval.params.catIds = [cat['id']]
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return tpr


val_json_name = glob.glob(os.path.join('xray_pred',f'result_{NAME}?.json')) + glob.glob(os.path.join('xray_pred',f'result_{NAME}??.json'))
# val_json_name = glob.glob(os.path.join('xray_pred/result_val?.json'))
print(val_json_name)
pred_all = []
#image_id和file_name对应
with open(os.path.join('xray_pred', 'newinstances_val2017.json'), 'r') as f:
    empty_test = json.load(f)
img_map = {}
for i in range(len(empty_test['images'])):
    img_map[empty_test['images'][i]['id']] = empty_test['images'][i]['file_name'].split('.')[0]

dict_class = {
        1:'Consolidation',
        2:'Fibrosis',
        3:'Effusion',
        4:'Nodule',
        5:'Mass',
        6:'Emphysema',
        7:'Calcification',
        8:'Atelectasis',
        9:'Fracture'
    }

#数据结构defaultdict(defaultdict(list))

ensembled_pred = defaultdict(lh_defaultlist)

with open('stat.txt','r') as f:
    stat=json.load(f)


for i in val_json_name:
    with open(i, 'r') as f:
        tmp = json.load(f)
        pred_all += tmp

for i in range(len(pred_all)):
    if pred_all[i]['category_id'] not in list(range(1, 10)):
        continue
    # if pred_all[i]['category_id'] == 6:
    cid = pred_all[i]['category_id']

    # if cid == 3:
    #     x, y, w, h = pred_all[i]['bbox']
    #     if (y + h) / 1024 < stat[dict_class[cid]]['loc'][0]:
    #         continue
    # if 
        

    # if cid==3:
    #     x, y, w, h = pred_all[i]['bbox']
    #     # if w < stat[dict_class[cid]]['width'][0] or w > stat[dict_class[cid]]['width'][1]:
    #     # if w<stat[dict_class[cid]]['width'][0]-20:
    #     #     continue
    #     # if h < stat[dict_class[cid]]['height'][0]-20:
    #     # if h < stat[dict_class[cid]]['height'][0] or h > stat[dict_class[cid]]['height'][1]:
    #         # continue
    #     # if w/h>stat[dict_class[cid]]['aspect'][1]:
    #     # if w / h < stat[dict_class[cid]]['aspect'][0] or w / h > stat[dict_class[cid]]['aspect'][1]:
    #     #     continue
    #     if  (y+h) / 1024 < stat[dict_class[cid]]['loc'][0]:
    #     # if y / 1024 < stat[dict_class[cid]]['loc'][0] or (y+h) / 1024 > stat[dict_class[cid]]['loc'][1] or x / 1024 < stat[dict_class[cid]]['loc'][2] or (x+w) / 1024 > stat[dict_class[cid]]['loc'][3]:
    #         continue
    pred_cur = pred_all[i]['bbox'] + [pred_all[i]['score']]
    img_id = pred_all[i]['image_id']
    cid = pred_all[i]['category_id']
    ensembled_pred[img_id][cid].append(pred_cur)
print('-' * 40 + '开始NMS测试' + '-' * 40)



VAL_GT = os.path.join('xray_pred','newinstances_val2017.json')
coco_gt = COCO(VAL_GT)
image_ids = coco_gt.getImgIds()
     
IOU_list = list(np.linspace(0, 0.5, 51))
# CONF_list = list(np.linspace(0, 0.5, 51))
CONF_list=[0,0.005]
# IOU_list = [0.5]
# CONF_list = [0.01]
# nms_style = ['OR', 'AND', 'MERGE','SOFT]
# iou_type = ['IoU', 'GIoU', 'DIoU', 'CIoU']
iou_type = ['DIoU','IoU']
nms_style = ['MERGE','OR']
want = {}
gt_nums = len(coco_gt.getAnnIds())
max_metric = 0
tmp_max = []

def nms_optimal(typ, sty):
    global max_metric
    global tmp_max
    # global max_lock
    for iou in IOU_list:
        for conf in CONF_list:
            tmp_ensembled_pred = Copy(ensembled_pred)
            print(f'开始检测type为{typ},style为{sty},iou阈值为{iou},conf阈值为{conf}下的结果')
            for k, v in tmp_ensembled_pred.items():
                for a, b in v.copy().items():
                    if len(b) != 0:
                        tmp_ensembled_pred[k][a] = lh_NMS(torch.tensor(b), conf_thresh=conf, iou_thresh=iou, style=sty ,type=typ)

            all_labels, nums = trans_2_standard_ann(tmp_ensembled_pred)
            print(f'预测框一共有{nums}个')
            if nums <= gt_nums:
                print('框太少不予检测(免得报错)')
                continue
            temp = _eval(coco_gt, image_ids, all_labels)
            # temp = _eval(coco_gt, image_ids, 'new_val.json')
            want[(typ, sty, iou, conf)] = temp
            print(f'type为{typ},style为{sty},iou阈值为{iou},conf阈值为{conf}下的指标为{temp}')
            
            # lock.acquire()
            if temp[-1] > max_metric:
                max_metric = temp[-1]
                tmp_max = [typ, sty, iou, conf] + temp
            print('当前最好成绩是type为{:^6},style为{:<5},iou阈值为{:.3f},conf阈值为{:.3f}下的指标为{}'.format(tmp_max[0],tmp_max[1],tmp_max[2],tmp_max[3],tmp_max[4:]))
            # lock.release()

# for typ in iou_type:
#     for sty in nms_style:
#         nms_optimal(typ,sty)

# thread_list = []
# max_lock = threading.Lock()

# for typ in iou_type:
#     for sty in nms_style:
#         thread_list.append(threading.Thread(target=nms_optimal,args=(typ, sty),name='{} with {}'.format(sty,typ)))
# for i in thread_list:
#     # i.setDaemon(False)
#     i.start()

# stat = 0

# while 1:
#     for i in thread_list:
#         stat = i.isAlive() | stat
#     if stat == True:
#         if len(tmp_max):
#             print('-'*100)
#             print('当前最好成绩是type为{:^6},style为{:<5},iou阈值为{:.3f},conf阈值为{:.3f}下的指标为{}'.format(tmp_max[0],tmp_max[1],tmp_max[2],tmp_max[3],tmp_max[4:]))
#             print('-'*100)

#     else:
#         final_tmp = []
#         imax = 0

#         for k, v in want.items():
#             if v[-1] > imax:
#                 imax= v[-1]
#                 final_tmp = [k[0], k[1], k[2], k[3]] + v
#             print(f'type为{k[0]},style为{k[1]},iou阈值为{k[2]},conf阈值为{k[3]}时的结果为{v}')
#             f.write(f'type为{k[0]},style为{k[1]},iou阈值为{k[2]},conf阈值为{k[3]}时的结果为{v}\n')

#         print('-'*40+'最佳'+'-'*40)    
#         print(f'最好的组合为type:{final_tmp[0]},style:{final_tmp[1]},iou阈值:{final_tmp[2]},conf阈值:{final_tmp[3]},结果是{final_tmp[3:]}')
#         f.write(f'\n\n最好的组合为type:{final_tmp[0]},style:{final_tmp[1]},iou阈值:{final_tmp[2]},conf阈值:{final_tmp[3]},结果是{final_tmp[3:]}')
#         nums=0
#         for i in ensembled_pred.keys():
#             nums += len(ensembled_pred[i][1])
#         print(nums)
#         print('-' * 40 + '结束' + '-' * 40)
#         break
final_tmp = []
imax = 0
# with open(os.path.join('xray_pred','all_eva.json'),'w') as f:
#     for k, v in want.items():
#         if v[-1] > imax:
#             imax= v[-1]
#             final_tmp = [k[0], k[1], k[2], k[3]] + v
#         print(f'type为{k[0]},style为{k[1]},iou阈值为{k[2]},conf阈值为{k[3]}时的结果为{v}')
#         f.write(f'type为{k[0]},style为{k[1]},iou阈值为{k[2]},conf阈值为{k[3]}时的结果为{v}\n')

#     print('-'*40+'最佳'+'-'*40)    
#     print(f'\n\n最好的组合为type:{final_tmp[0]},style:{final_tmp[1]},iou阈值:{final_tmp[2]},conf阈值:{final_tmp[3]},结果是{final_tmp[3:]}')
#     f.write(f'\n\n最好的组合为type:{final_tmp[0]},style:{final_tmp[1]},iou阈值:{final_tmp[2]},conf阈值:{final_tmp[3]},结果是{final_tmp[3:]}')

print('-' * 40 + '结束' + '-' * 40) 

def single_pred(typ, sty, iou, conf):
    tmp_ensembled_pred = Copy(ensembled_pred)
    print(f'开始检测type为{typ},style为{sty},iou阈值为{iou},conf阈值为{conf}下的结果')
    for k, v in tmp_ensembled_pred.items():
        for a, b in v.copy().items():
            if len(b) != 0:
                tmp_ensembled_pred[k][a] = lh_NMS(torch.tensor(b), conf_thresh=conf, iou_thresh=iou, style=sty ,type=typ)

    all_labels, nums = trans_2_standard_ann(tmp_ensembled_pred, 1)
    print(f'预测框一共有{nums}个')
    temp = _eval(coco_gt, image_ids, all_labels)
    print(f'type为{typ},style为{sty},iou阈值为{iou},conf阈值为{conf}下的指标为{temp}')
    # temp = _eval(coco_gt, image_ids,f'new_{NAME}.json')
    # print(f'type为{typ},style为{sty},iou阈值为{iou},conf阈值为{conf}下的指标为{temp}')

single_pred('DIoU','MERGE',0.170,0.00)
single_pred('DIoU','MERGE',0.250,0.00)
single_pred('DIoU','MERGE',0.250,0.005)

def output_test(typ, sty, iou, conf):
    tmp_ensembled_pred = Copy(ensembled_pred)
    for k, v in tmp_ensembled_pred.items():
        for a, b in v.copy().items():
            if len(b) != 0:
                tmp_ensembled_pred[k][a] = lh_NMS(torch.tensor(b), conf_thresh=conf, iou_thresh=iou, style=sty, type=typ)
    all_labels, nums = trans_2_standard_ann(tmp_ensembled_pred, 1)
    print(f'预测框一共有{nums}个')

# output_test('DIoU', 'MERGE', 0.170,0)
