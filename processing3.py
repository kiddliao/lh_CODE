import os
import json
import cv2
import random
import re
import time
import shutil
from functools import cmp_to_key
import tensorboardX
#查看数据集的样本种类的个数并打印偏移gt框
category={1:'person',2:'bicycle',3:'car'}
f1 = open(os.path.join('coco', 'annotations', 'newinstances_train2017.json'), 'r')
f2 = open(os.path.join('coco', 'annotations', 'newinstances_test2017.json'), 'r')
f3 = open(os.path.join('coco', 'annotations', 'newinstances_val2017.json'), 'r')
#train.json和val.json的id都是从0开始 得做处理
label1 = json.load(f1)
label2 = json.load(f2)
label3 = json.load(f3)
for i in range(len(label1['annotations'])):
    label1['annotations'][i]['image_id'] += 1366
for i in range(len(label1['images'])):
    label1['images'][i]['id'] += 1366
f1.close()
f2.close()
f3.close()
label = {}
label['images'] = label1['images'] + label2['images'] + label3['images']
label['annotations'] = label1['annotations'] + label2['annotations'] + label3['annotations']
label['info'] = label1['info']
label['licenses'] = label1['licenses']
label['categories'] = label1['categories']
# 遍历每个类别的样本数量
def print_count(res):
    person_count, bike_count, car_count = 0, 0, 0
    for i in range(len(res['annotations'])):
        id = res['annotations'][i]['category_id']
        if id == 1:
            person_count += 1
        elif id == 2:
            bike_count += 1
        elif id == 3:
            car_count += 1
        else: raise TypeError
    print(f'person类总共有{person_count}个样本')
    print(f'bicycle类总共有{bike_count}个样本')
    print(f'car类总共有{car_count}个样本')
print('---------训练集----------')
print_count(label1)
print('---------测试集----------')
print_count(label2)
print('---------验证集----------')
print_count(label3)
print('---------数据集----------')
print_count(label)

# 随机找五张图片画出框和10个像素偏移的框
with open(os.path.join('coco', 'annotations', 'newinstances_train2017.json'), 'r') as f:
    label = json.load(f)
    images = label['images']
    categories = label['categories']
    annotations = label['annotations']
    random.shuffle(images)
    for img in images[:10]:
        bbox = []
        cid = 0
        img_name = img['file_name']
        img_id = img['id']
        flag=0
        for res in annotations:
            if res['image_id'] != img_id and flag>0:break
            if res['image_id'] == img_id:
                flag += 1
                bbox.append([*res['bbox'], res['category_id']])
        pic = cv2.imread(os.path.join('coco', 'train2017', img_name), 0)
        for box in bbox:
            x, y, w, h, id = box
            cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)  #2是线的宽度
            cv2.rectangle(pic, (x+10, y+10), (x + w+10, y + h+10), (0, 255, 0), 2)  #2是线的宽度
            cv2.putText(pic, '{}, {:.3f}'.format(category[id], 1),
                        (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        cv2.namedWindow(img_name)
        cv2.imshow(img_name, pic)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

        
