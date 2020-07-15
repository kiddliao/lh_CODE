import os
import glob
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from torchvision import transforms
#实现图片读取的数据管道
os.chdir(os.path.join('..','REMOTE','datasets','coco_mocod','obstruct'))
class FlirDataset(Dataset):
    def __init__(self, root_dir, set_name, cal_mean_std=False, mean_std_path=None, transform=None):
        self.set_name = set_name
        self.root_dir = root_dir
        # self.img_path = glob.glob(os.path.join('coco',self.set_name,'*.jpg'))
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, 'newinstances_' + self.set_name + '.json'))
        # self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'newinstances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()  #获取image在标注文件里的id 0-n label["images"][0]["image_id"]
        self.load_classes()
        self.cal_mean_std = cal_mean_std
        self.mean_std_path = mean_std_path
        if self.cal_mean_std:
            self.means = [0, 0, 0]
            self.stdevs = [0, 0, 0]
            self.get_mean_std(self.mean_std_path)
    
    def get_mean_std(self,mean_std_path):
        for i in range(len(self.image_ids)):
            img = self.load_image(i)
            img = img.transpose(2,0,1) #hwc 2 chw
            for j in range(3):
                self.means[j] += img[j, :, :].mean()
                self.stdevs[j] += img[j, :, :].std()

        self.means = np.asarray(self.means) / len(self.image_ids)
        self.stdevs = np.asarray(self.stdevs) / len(self.image_ids)
        print("{} : normMean = {}".format(type, self.means))
        print("{} : normstdevs = {}".format(type, self.stdevs))
        
        # 将得到的均值和标准差写到文件中，之后就能够从中读取
        with open(mean_std_path, 'w') as f:
            json.dump({'means': list(self.means), 'stdevs': list(self.stdevs)}, f)

    def load_classes(self):
        #coco.getCatIds()返回的是class分类在label["categories"]里的id 一般是0-80
        #coco.loadCats()返回列表,里面放的是label["categories"][i]
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels = {}
        # 如果自己数据集标签只有四类1,2,3,17 
        # coco_labels和coco_label_inverse将其重置为0,1,2,3 
        # 这样输出层就只有4个神经元而不是80个
        self.coco_labels_inverse = {} 
        self.labels = {}
        for i in range(len(categories)):
            self.coco_labels[i] = categories[i]['id']
            self.coco_labels_inverse[categories[i]['id']] = i
            self.classes[categories[i]['name']] = i
            self.labels[i] = categories[i]['name']

    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, index):
        img = self.load_image(index)
        annot = self.load_annotations(index)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample
    def load_image(self, image_index):
        image_name = self.coco.loadImgs(self.image_ids[image_index])[0]  #loadImgs是返回一个列表 里面放的是label["images"][i]
        # image_path = os.path.join(self.root_dir, 'train', image_name['file_name'].replace('jpg','png'))
        image_path = os.path.join(self.root_dir, 'JPEGImages', image_name['file_name'])
        # image_path = os.path.join(self.root_dir, self.set_name, image_name['file_name'])
        #三通道图片
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)





        # cv2.imwrite(osp.join(path, 'test_cat.jpg'), img)
        #单通道图片
        # img = cv2.imread(image_path, 0)
        # img=img.reshape((*img.shape,1))
        return img.astype(np.float32) / 255.
    def load_annotations(self, image_index):
        # coco.getAnnIds()返回label["annotations"]里'image_id'是i的所有标注的"id"
        annotaions_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        if len(annotaions_ids) == 0:
            return annotations
        coco_annotations = self.coco.loadAnns(annotaions_ids)
        for idx, item in enumerate(coco_annotations):
            #忽略没有宽或高的标注
            if item['bbox'][2] < 1 or item['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0,:4] = item['bbox']
            annotation[0, 4] = self.coco_label_to_label(item['category_id'])
            annotations=np.append(annotations,annotation,axis=0)
        
        #transform from [x,y,w,h](左上角坐标) to [x1,y1,x2,y2](左上角坐标和右下角坐标)
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations
    
    def coco_label_to_label(self, coco_label):
        # 将coco_label [1,2,3,17] 转换为label [0,1,2,3]
        return self.coco_labels_inverse[coco_label]
    def label_to_coco_label(self, label):
        # 将label [0,1,2,3] 转换为coco_label [1,2,3,17]
        return self.coco_labels[label]

def collater(batch):
    '''
    自定义数据集读取方法
    '''
    imgs, annots, scales = [], [], []
    for s in batch:
        imgs.append(s['img'])
        annots.append(s['annot'])
        scales.append(s['scale'])
    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx,:annot.shape[0],:] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    imgs = imgs.permute(0, 3, 1, 2) #opencv read nHWc , convert to ncHW
    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    def __init__(self, img_size=512):
        self.img_size = img_size
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size
        
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        #三通道图
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image #注意这里没有居中直接把新图放到了左上角
        #单通道图
        # new_image = np.zeros((self.img_size, self.img_size))
        # new_image[0:resized_height, 0:resized_width] = image
        # new_image = new_image.reshape((*(new_image.shape), 1))      
        annots[:,:4] *= scale  #coco里的左上角和右下角的坐标代表距离图片左上顶点的距离 所以直接乘上scale
        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}

class Normalizer(object):
    def __init__(self, mean=[0.526, 0.526, 0.526], std=[0.245, 0.245, 0.245], mean_std_path=None):
        self.mean = None
        self.std = None
        self.mean_std_path = mean_std_path
        if self.mean_std_path:
            with open(self.mean_std_path, 'r') as f:
                mean_std = json.load(f)
                self.mean = np.array(mean_std['means'])
                self.std = np.array(mean_std['stdevs'])
        else:
            self.mean = np.array(mean)
            self.std = np.array(std)
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:,::-1,:]
            h, w, c = image.shape
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            annots[:, 0] = w - x2
            annots[:, 2] = w - x1
            sample = {'img': image, 'annot': annots}
        return sample

            
                
            
    
        
            
        
    








        

        
        

        