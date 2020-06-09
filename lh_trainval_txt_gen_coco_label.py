import os
import json
import cv2
import time
from functools import cmp_to_key
import random
#把train.txt转换为coco标注文件
os.chdir(os.path.join('..', 'datasets', 'coco_mocod','obstruct'))
os.makedirs('annotations', exist_ok=True)
def convert(name):
    with open(os.path.join(f"2007_{name}.txt"), 'r') as f:
        res = f.readlines()
        res = list(map(lambda x: x.strip().split('/')[-1].split('.')[0], res))
    label = {}
    label["categories"]=[{
        "name": "Car",
        "id": 1,
        "supercategory": "unknown"
        },
        {
        "name": "Ship",
        "id": 2,
        "supercategory": "unknown"
        },
        {
        "name": "Plane",
        "id": 3,
        "supercategory": "unknown"
        },
        {
        "name": "Human",
        "id": 4,
        "supercategory": "unknown"
        }]
    label["info"] = {}
    label["licenses"] = []
    label['images'] = []
    label['annotations'] = []

    anno_sample = {
        "image_id": 0,
        "extra_info": {
            "human_annotated": True
        },
        "category_id": 0,
        "iscrowd": 0,
        "id": 0,
        "segmentation": [],
        "bbox": [],
        "area": 0
        }
    img_sample={
        "extra_info": {},
        "subdirs": ".",
        "id": 0,
        "width": 0,
        "file_name": "",
        "height": 0
        }
    img_id = 0
    anno_id = 1
    for i in range(len(res)):
        image = img_sample.copy()
        image["id"] = img_id
        image["file_name"]=res[i]+'.jpg'
        pic = cv2.imread(os.path.join('JPEGImages', res[i]+'.jpg'))
        h_, w_, _ = pic.shape
        image['width'], image['height'] = w_, h_
        label['images'].append(image.copy())
        with open(os.path.join('labels', res[i] + '.txt'), 'r') as f:
            ans = f.readlines()
        for j in range(len(ans)):
            cid, xc, yc, w, h = list(map(lambda x:float(x),ans[j].strip().split()))
            xc, yc, w, h = xc * w_, yc * h_, w * w_, h * h_
            annotation = anno_sample.copy()
            annotation['image_id'] = img_id
            annotation['category_id'] = int(cid) + 1
            annotation['id'] = anno_id
            x, y, w, h = xc - w / 2, yc - h / 2, w, h
            annotation['bbox'] = [x, y, w, h]
            annotation['area'] = w * h
            label['annotations'].append(annotation.copy())
            anno_id += 1
        img_id += 1
    with open(os.path.join('annotations', f'newinstances_{name}2017.json'), 'w') as f:
        json.dump(label, f)


convert('val')
convert('train')
# convert('test')