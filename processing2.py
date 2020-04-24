import os
import json
import random
import re
import time
import shutil
from functools import cmp_to_key
f1 = open(os.path.join('coco', 'annotations', 'instances_train2017.json'), 'r')
f2 = open(os.path.join('coco', 'annotations', 'instances_test2017.json'), 'r')
f3 = open(os.path.join('coco', 'annotations', 'instances_val2017.json'), 'r')
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
#处理图片名字
name = re.compile('\d{5}\.jpg')
for i in range(len(label['images'])):
    res = name.search(label['images'][i]['file_name'])
    label['images'][i]['file_name'] = res.group()

trainp, testvalp = 0.8, 0.2
testp, valp = 0.5, 0.5
annotations = label['annotations']
images = label['images']
n = len(images)
trainnum = int(n * trainp)
testvalnum = n - trainnum
testnum = int(testvalnum * testp)
valnum = testvalnum - testnum

random.seed(0)
random.shuffle(images)
train, val, test = {}, {}, {}
train['info'] = test['info'] = val['info'] = label1['info']
train['licenses'] = test['licenses'] = val['licenses'] = label1['licenses']
train['categories'] = test['categories'] = val['categories'] = label1['categories']
train['images'] = images[:trainnum]
test['images'] = images[trainnum:trainnum + testnum]
val['images'] = images[trainnum + testnum:]
val['annotations'], test['annotations'], train['annotations'] = [], [], []
trainid, valid, testid = [], [], []
for i in range(len(train['images'])):
    trainid.append(train['images'][i]['id'])
for i in range(len(test['images'])):
    testid.append(test['images'][i]['id'])
for i in range(len(val['images'])):
    valid.append(val['images'][i]['id'])
for i in range(len(annotations)):
    id = annotations[i]['image_id']
    if id in trainid:
        train['annotations'].append(annotations[i])
    elif id in testid:
        test['annotations'].append(annotations[i])
    elif id in valid:
        val['annotations'].append(annotations[i])
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
        res['annotations'][i]['id'] = i
    for i in range(len(res['images'])):
        id = res['images'][i]['id']
        res['images'][i]['id'] = i
        for j in range(len(res['annotations'])):
            if res['annotations'][j]['image_id'] == id:
                res['annotations'][j]['image_id'] = i
    end = time.time()
    print(f'处理用时{end-start}s')

process(train)
process(test)
process(val)


#把val和test的图片移到对应文件夹
trainpath = os.path.join('coco', 'train2017')
testpath = os.path.join('coco', 'test2017')
valpath = os.path.join('coco', 'val2017')

for i in range(len(test['images'])):
    file_name = test['images'][i]['file_name']
    shutil.move(os.path.join(trainpath, file_name), testpath)
    print(f'{file_name}从train2017移动到test2017')
for i in range(len(val['images'])):
    file_name = val['images'][i]['file_name']
    shutil.move(os.path.join(trainpath, file_name), valpath)
    print(f'{file_name}从train2017移动到val2017')

with open(os.path.join('coco', 'annotations', 'newinstances_train2017.json'), 'w') as f:
    json.dump(train,f)
with open(os.path.join('coco', 'annotations', 'newinstances_test2017.json'), 'w') as f:
    json.dump(test, f)
with open(os.path.join('coco', 'annotations', 'newinstances_val2017.json'), 'w') as f:
    json.dump(val,f)

     


