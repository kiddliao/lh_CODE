import os
import json
import cv2
import time
from functools import cmp_to_key
import random
#csig比赛把比赛给的标签格式转换成coco
# os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray'))
os.makedirs('annotations', exist_ok=True)

with open(os.path.join("train_x-ray.json"), 'r') as f:
    raw = json.load(f)
label = {}
category1={"肺实变":"Consolidation",
"纤维化表现":"Fibrosis",
"胸腔积液":"Effusion",
"结节":"Nodule",
"肿块":"Mass",
"肺气肿":"Emphysema",
"钙化":"Calcification",
"肺不张":"Atelectasis",
"骨折":"Fracture"
}
category2={"Consolidation":1,
"Fibrosis":2,
"Effusion":3,
"Nodule":4,
"Mass":5,
"Emphysema":6,
"Calcification":7,
"Atelectasis":8,
"Fracture":9
}
category3={"Consolidation":0,
"Fibrosis":0,
"Effusion":0,
"Nodule":0,
"Mass":0,
"Emphysema":0,
"Calcification":0,
"Atelectasis":0,
"Fracture":0
}
category4={1:"Consolidation",
2:"Fibrosis",
3:"Effusion",
4:"Nodule",
5:"Mass",
6:"Emphysema",
7:"Calcification",
8:"Atelectasis",
9:"Fracture"
}

label["categories"]=[{
      "name": "Consolidation",
      "id": 1,
      "supercategory": "unknown"
    },
    {
      "name": "Fibrosis",
      "id": 2,
      "supercategory": "unknown"
    },
    {
      "name": "Effusion",
      "id": 3,
      "supercategory": "unknown"
    },
    {
      "name": "Nodule",
      "id": 4,
      "supercategory": "unknown"
    },
    {
      "name": "Mass",
      "id": 5,
      "supercategory": "unknown"
    },
    {
      "name": "Emphysema",
      "id": 6,
      "supercategory": "unknown"
    },
    {
      "name": "Calcification",
      "id": 7,
      "supercategory": "unknown"
    },
    {
      "name": "Atelectasis",
      "id": 8,
      "supercategory": "unknown"
    },
    {
      "name": "Fracture",
      "id": 9,
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
anno_id = 0
for i in range(len(raw)):
    if raw[i]['syms'] == []: continue
    image = img_sample.copy()
    image["id"] = img_id
    image["file_name"]=raw[i]["file_name"].replace('.png','.jpg')
    pic = cv2.imread(os.path.join('train', raw[i]["file_name"]), 0)
    h, w = pic.shape
    image['width'], image['height'] = w, h
    label['images'].append(image.copy())
    for j in range(len(raw[i]["syms"])):
        annotation = anno_sample.copy()
        annotation['image_id'] = img_id
        annotation['category_id'] = category2[raw[i]['syms'][j]]
        annotation['id'] = anno_id
        x1, y1, x2, y2 = raw[i]['boxes'][j]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        annotation['bbox'] = [x, y, w, h]
        annotation['area'] = w * h
        label['annotations'].append(annotation.copy())
        anno_id += 1
    img_id += 1

trainp, testvalp = 0.85, 0.15
# testp, valp = 0.5, 0.5
annotations = label['annotations']
images = label['images']
n = len(images)
trainnum = int(n * trainp)
# testvalnum = n - trainnum
# testnum = int(testvalnum * testp)
# valnum = testvalnum - testnum
valnum = n - trainnum


random.seed(0)
random.shuffle(label['images'])
# train, val, test = {}, {}, {}
train, val = {}, {}
train['info'] = val['info'] = label['info']
train['licenses'] = val['licenses'] = label['licenses']
train['categories'] = val['categories'] = label['categories']
train['images'] = label['images'][:trainnum]
val['images'] = label['images'][trainnum:]
val['annotations'], train['annotations'] = [], []
trainid, valid = [], []
for i in range(len(train['images'])):
    trainid.append(train['images'][i]['id'])
for i in range(len(val['images'])):
    valid.append(val['images'][i]['id'])

print(f'训练集有{len(trainid)}张,验证集有{len(valid)}张')
for i in range(len(label['annotations'])):
    id = label['annotations'][i]['image_id']
        
    if id in trainid:
        train['annotations'].append(label['annotations'][i])
    elif id in valid:
        val['annotations'].append(label['annotations'][i])
    else:
        raise TypeError

def compare1(a, b):
    if a['id'] > b['id']:
        return 1
    else: return - 1
def compare2(a, b):
    if a['image_id'] > b['image_id']:
        return 1
    else: return - 1
# 把image_id从0开始
def process(res):
    start = time.time()
    res['images'].sort(key=cmp_to_key(compare1))
    res['annotations'].sort(key=cmp_to_key(compare2))
    for i in range(len(res['annotations'])):
        res['annotations'][i]['id'] = i +1 
    # 按道理这里是可以O(n)一次遍历的 已经排好序了 顺着给每个相同image_id图片的分组赋值1 1 1 2 2...n即可
    for i in range(len(res['images'])):
        id = res['images'][i]['id']
        res['images'][i]['id'] = i
        flag = 0
        for j in range(len(res['annotations'])):#符合条件的都是连在一起的
            if res['annotations'][j]['image_id'] == id:
                res['annotations'][j]['image_id'] = i
                flag += 1
            elif res['annotations'][j]['image_id'] != id and flag > 0:
                break
        flag = 0
    end = time.time()
    print(f'处理用时{end-start}s')

process(val)
process(train)


with open(os.path.join('annotations', 'newinstances_train2017.json'), 'w') as f:
    json.dump(train,f)
with open(os.path.join('annotations', 'newinstances_val2017.json'), 'w') as f:
    json.dump(val,f)



label1 = train
label2 = val
# 遍历每个类别的样本数量
def print_count(res):
    categoryx=category3.copy()
    for i in range(len(res['annotations'])):
        id = res['annotations'][i]['category_id']
        categoryx[category4[id]] += 1
    for k, v in categoryx.items():
        print(f"{k}有{v}张")
        
print('---------训练集----------')
print_count(train)
print('---------验证集----------')
print_count(val)
print('---------数据集----------')
print_count(label)
os.chdir('train')
os.system('rename png jpg *')
    
        
    
    