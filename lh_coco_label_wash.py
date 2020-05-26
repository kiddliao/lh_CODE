import os
import json
import random
import re
import time
import shutil
from functools import cmp_to_key
#删除coco数据集里样本比较小的类 不检测
# os.chdir(os.path.join('..','REMOTE','datasets','coco_FLIR'))
f1 = open(os.path.join('coco', 'annotations', 'instances_train2017.json'), 'r')
f2 = open(os.path.join('coco', 'annotations', 'instances_val2017.json'), 'r')
label1 = json.load(f1)
label2 = json.load(f2)
cate_sample = [{'id':1, 'name':'person', 'supercategory':'unknown'},
               {'id': 2, 'name': 'bicycle', 'supercategory': 'unknown'},
               {'id': 3, 'name': 'car', 'supercategory': 'unknown'}]
label1['categories'] = label2['categories'] = cate_sample
dog_count = 0

def wash_labels(labels):
    global dog_count
    for i in range(len(labels['images'])):
        file_name = labels['images'][i]['file_name']
        labels['images'][i]['file_name'] = file_name.strip().split('/')[-1].split('.')[0] + '.jpg'
    #删除不用的标注并且重排annotations的id
    newanno = []
    for i in range(len(labels['annotations'])):
        if labels['annotations'][i]['category_id'] == 17:
            dog_count += 1
            print(f'{dog_count} dog deleted')
        else:
            newanno.append(labels['annotations'][i])
    labels['annotations'] = newanno.copy()
    for i in range(len(labels['annotations'])):
        labels['annotations'][i]['id'] = i + 1
        
wash_labels(label1)
wash_labels(label2)


with open(os.path.join('coco', 'annotations', 'newinstances_train2017.json'), 'w') as f:
    json.dump(label1,f)

with open(os.path.join('coco', 'annotations', 'newinstances_val2017.json'), 'w') as f:
    json.dump(label2,f)

label = {}
label['images'] = label1['images'] + label2['images'] 
label['annotations'] = label1['annotations'] + label2['annotations']
label['info'] = label1['info']
label['licenses'] = label1['licenses']
label['categories'] = label1['categories']

category={1:'person',2:'bicycle',3:'car'}
# 遍历每个类别的样本数量
def print_count(res):
    person_count, bike_count, car_count, dog_count = 0, 0, 0, 0
    for i in range(len(res['annotations'])):
        id = res['annotations'][i]['category_id']
        if id == 1:
            person_count += 1
        elif id == 2:
            bike_count += 1
        elif id == 3:
            car_count += 1
        else:
            print('异常id是', id, f'是第{i}项')
            raise TypeError
    print(f'person类总共有{person_count}个样本')
    print(f'bicycle类总共有{bike_count}个样本')
    print(f'car类总共有{car_count}个样本')
print('---------训练集----------')
print_count(label1)
print('---------验证集----------')
print_count(label2)
print('---------数据集----------')
print_count(label)


    