import cv2
import os
import json
import random
import shutil
import time
import glob
import xml.etree.ElementTree as ET
from functools import cmp_to_key
from collections import defaultdict
#终极脚本 xml的voc数据集处理成coco格式
#1划分训练集测试集验证集
#2标注转换为coco
#3统计
classes = {
    'knife': 1,
    'scissors': 2,
    'lighter': 3,
    'zippooil': 4,
    'pressure': 5,
    'slingshot': 6,
    'handcuffs': 7,
    'nailpolish': 8,
    'powerbank': 9,
    'firecrackers': 10
}
rev_classes = {
    1: 'knife',
    2: 'scissors',
    3: 'lighter',
    4: 'zippooil',
    5: 'pressure',
    6: 'slingshot',
    7: 'handcuffs',
    8: 'nailpolish',
    9: 'powerbank',
    10: 'firecrackers'
}
# os.chdir(os.path.join('..','REMOTE', 'datasets', 'voc_infrared'))
#读取xml并转换为annotations.json
xmlpaths = glob.glob(os.path.join('Annotations', '*.xml'))
xmlpaths.sort()
label = {}
image_id = 0
annotation_id = 0
label['info'], label['licenses'], label['images'], label['annotations'], label['categories'] = [], [], [], [], []

for i in range(len(xmlpaths)):
    tree = ET.parse(xmlpaths[i])
    root = tree.getroot()
    images = {}
    images['extra_info'] = {}
    images['subdirs'] = '.'
    images['id'] = image_id
    images['width'] = int(root.find('size').find('width').text)
    images['file_name'] = root.find('filename').text.strip().split('/')[-1]
    images['height'] = int(root.find('size').find('height').text)
    label['images'].append(images.copy())
    for obj in root.iter('object'):
        cls = obj.find('name').text.lower()
        cls_id = classes[cls]
        xmlbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                                  float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        annotations = {}
        annotations['image_id'] = image_id
        annotations['extra_info'] = {'human_annotated': True}.copy()
        annotations['category_id'] = cls_id
        annotations['iscrowd'] = 0
        annotations['id'] = annotation_id
        annotations['segmentation'] = []
        annotations['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
        annotations['area'] = (xmax - xmin) * (ymax - ymin)
        label['annotations'].append(annotations.copy())
        annotation_id += 1
    image_id += 1
for k, v in classes.items():
    categories = {}
    categories['name'] = k
    categories['id'] = v
    categories['supercategory'] = 'unknown'
    label['categories'].append(categories.copy())

#划分训练集验证集测试集
# trainp, testvalp = 0.8, 0.2
# testp, valp = 0.5, 0.5
# annotations = label['annotations']
# images = label['images']
# n = len(images)
# trainnum = int(n * trainp)
# testvalnum = n - trainnum
# testnum = int(testvalnum * testp)
# valnum = testvalnum - testnum

# random.seed(0)
# random.shuffle(images)
# train, val, test = {}, {}, {}
# train['info'] = test['info'] = val['info'] = label['info']
# train['licenses'] = test['licenses'] = val['licenses'] = label['licenses']
# train['categories'] = test['categories'] = val['categories'] = label['categories']
# train['images'] = images[:trainnum]
# test['images'] = images[trainnum:trainnum + testnum]
# val['images'] = images[trainnum + testnum:]
# val['annotations'], test['annotations'], train['annotations'] = [], [], []
# trainid, valid, testid = [], [], []

#只划分验证集和测试集
trainp, valp = 0.9, 0.1
annotations = label['annotations']
images = label['images']
n = len(images)
trainnum = int(n * trainp)
valnum = n - trainnum

random.seed(1)
random.shuffle(images)
train, val = {}, {}
train['info'] = val['info'] = label['info']
train['licenses'] = val['licenses'] = label['licenses']
train['categories'] = val['categories'] = label['categories']
train['images'] = images[:trainnum]
val['images'] = images[trainnum:]
val['annotations'], train['annotations'] = [], []
trainid, valid = [], []

for i in range(len(train['images'])):
    trainid.append(train['images'][i]['id'])
# for i in range(len(test['images'])):
#     testid.append(test['images'][i]['id'])
for i in range(len(val['images'])):
    valid.append(val['images'][i]['id'])
print(f'训练集有{len(trainid)}张,验证集有{len(valid)}张')
# print(f'训练集有{len(trainid)}张,验证集有{len(valid)}张,测试集有{len(testid)}张')
for i in range(len(annotations)):
    id = annotations[i]['image_id']
    if id in trainid:
        train['annotations'].append(annotations[i])
    # elif id in testid:
    #     test['annotations'].append(annotations[i])
    elif id in valid:
        val['annotations'].append(annotations[i])
    else:
        raise TypeError


def compare1(a, b):
    if a['id'] > b['id']:
        return 1
    else:
        return -1


def compare2(a, b):
    if a['image_id'] > b['image_id']:
        return 1
    else:
        return -1


# 把image_id从0开始
def process(res):
    start = time.time()
    res['images'].sort(key=cmp_to_key(compare1))
    res['annotations'].sort(key=cmp_to_key(compare2))
    for i in range(len(res['annotations'])):
        res['annotations'][i]['id'] = i + 1
    for i in range(len(res['images'])):
        id = res['images'][i]['id']
        res['images'][i]['id'] = i
        for j in range(len(res['annotations'])):
            if res['annotations'][j]['image_id'] == id:
                res['annotations'][j]['image_id'] = i
    end = time.time()
    print(f'处理用时{end-start}s')


process(train)
# process(test)
process(val)

# #把val和test的图片移到对应文件夹
# trainpath = os.path.join('coco', 'train2017')
# testpath = os.path.join('coco', 'test2017')
# valpath = os.path.join('coco', 'val2017')

# for i in range(len(test['images'])):
#     file_name = test['images'][i]['file_name']
#     shutil.move(os.path.join(trainpath, file_name), testpath)
#     print(f'{file_name}从train2017移动到test2017')
# for i in range(len(val['images'])):
#     file_name = val['images'][i]['file_name']
#     shutil.move(os.path.join(trainpath, file_name), valpath)
#     print(f'{file_name}从train2017移动到val2017')
os.makedirs('annotations', exist_ok=True)
with open(os.path.join('annotations', 'instances_train2017.json'), 'w') as f:
    json.dump(train, f)
# with open(os.path.join('annotations', 'instances_test2017.json'), 'w') as f:
#     json.dump(test, f)
with open(os.path.join('annotations', 'instances_val2017.json'), 'w') as f:
    json.dump(val, f)


def print_count(res):
    stat = defaultdict(int)
    for i in range(len(res['annotations'])):
        id = res['annotations'][i]['category_id']
        stat[rev_classes[id]] += 1
    print(sorted(stat.items(), key=lambda x: x[0]))


print('---------训练集----------')
print_count(train)
# print('---------测试集----------')
# print_count(test)
print('---------验证集----------')
print_count(val)
print('---------数据集----------')
print_count(label)

# #把png改为jpg
# files = os.listdir(os.path.join("JPEGImages"))
# for filename in files:
#     portion = os.path.splitext(filename)#portion为名称和后缀分离后的列表
#     if portion[1] != '.jpg':
#         newname = portion[0]+".jpg"
#         # print(filename,'-->',newname)
#         os.rename(os.path.join('JPEGImages',filename),os.path.join('JPEGImages',newname))