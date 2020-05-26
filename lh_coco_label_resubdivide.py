import os
import json
import random
import re
import time
import shutil
from functools import cmp_to_key
#对coco格式的数据集重新划分
os.chdir(os.path.join('..','REMOTE','datasets','coco_FLIR'))
#修改图片后缀
# files = os.listdir(os.path.join("coco",'train2017'))
# for filename in files:
#     portion = os.path.splitext(filename)#portion为名称和后缀分离后的列表
#     if portion[1] != '.jpg':
#         newname = portion[0]+".jpg"
#         # print(filename,'-->',newname)
#         os.rename(os.path.join('coco','train2017',filename),os.path.join('coco','train2017',newname))



f1 = open(os.path.join('coco', 'annotations', 'instances_train2017.json'), 'r')
f2 = open(os.path.join('coco', 'annotations', 'instances_test2017.json'), 'r')
f3 = open(os.path.join('coco', 'annotations', 'instances_val2017.json'), 'r')
#train.json和val.json的id都是从0开始 得做处理
label1 = json.load(f1)
label2 = json.load(f2)
label3 = json.load(f3)
n = label1['images'][-1]['id'] + 1
m = label2['images'][-1]['id'] + n + 1
#测试集
for i in range(len(label2['annotations'])):
    label2['annotations'][i]['image_id'] += n
for i in range(len(label2['images'])):
    label2['images'][i]['id'] += n
#验证集
for i in range(len(label3['annotations'])):
    label3['annotations'][i]['image_id'] += m
for i in range(len(label3['images'])):
    label3['images'][i]['id'] += m
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
# name = re.compile('\d{5}\.jpg')
# for i in range(len(label['images'])):
#     res = name.search(label['images'][i]['file_name'])
#     label['images'][i]['file_name'] = res.group()
for i in range(len(label['images'])):
    tmp = label['images'][i]['file_name']
    tmp = tmp.strip().split('/')[-1]
    tmp = tmp.split('.')[0] + '.jpg'
    label['images'][i]['file_name'] = tmp


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
random.shuffle(label['images'])
train, val, test = {}, {}, {}
train['info'] = test['info'] = val['info'] = label1['info']
train['licenses'] = test['licenses'] = val['licenses'] = label1['licenses']
lc = label1['categories'][:4]
lc[3]['name'], lc[3]['id'] = 'dog', 4
train['categories'] = test['categories'] = val['categories'] = lc
train['images'] = label['images'][:trainnum]
test['images'] = label['images'][trainnum:trainnum + testnum]
val['images'] = label['images'][trainnum + testnum:]
val['annotations'], test['annotations'], train['annotations'] = [], [], []
trainid, valid, testid = [], [], []
for i in range(len(train['images'])):
    trainid.append(train['images'][i]['id'])
for i in range(len(test['images'])):
    testid.append(test['images'][i]['id'])
for i in range(len(val['images'])):
    valid.append(val['images'][i]['id'])

print(f'训练集有{len(trainid)}张,测试集有{len(testid)}张,验证集有{len(valid)}张')
for i in range(len(label['annotations'])):
    id = label['annotations'][i]['image_id']
    #把dog转换为4
    if label['annotations'][i]['category_id'] == 17:
        label['annotations'][i]['category_id'] = 4
        
    if id in trainid:
        train['annotations'].append(label['annotations'][i])
    elif id in testid:
        test['annotations'].append(label['annotations'][i])
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
        res['annotations'][i]['id'] = i + 1
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


process(test)
process(val)
process(train)

#把val和test的图片移到对应文件夹
trainpath = os.path.join('coco', 'train2017')
testpath = os.path.join('coco', 'test2017')
valpath = os.path.join('coco', 'val2017')

for i in range(len(test['images'])):
    file_name = test['images'][i]['file_name']
    shutil.move(os.path.join(trainpath, file_name), testpath)
    # print(f'{file_name}从train2017移动到test2017')
for i in range(len(val['images'])):
    file_name = val['images'][i]['file_name']
    shutil.move(os.path.join(trainpath, file_name), valpath)
    # print(f'{file_name}从train2017移动到val2017')

with open(os.path.join('coco', 'annotations', 'newinstances_train2017.json'), 'w') as f:
    json.dump(train,f)
with open(os.path.join('coco', 'annotations', 'newinstances_test2017.json'), 'w') as f:
    json.dump(test, f)
with open(os.path.join('coco', 'annotations', 'newinstances_val2017.json'), 'w') as f:
    json.dump(val,f)



category={1:'person',2:'bicycle',3:'car',4:'dog'}
label1 = train
label2 = test
label3 = val
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
        elif id == 4:
            dog_count += 1
        else:
            print('异常id是',id)
            raise TypeError
    print(f'person类总共有{person_count}个样本')
    print(f'bicycle类总共有{bike_count}个样本')
    print(f'car类总共有{car_count}个样本')
    print(f'dog类总共有{dog_count}个样本')
print('---------训练集----------')
print_count(train)
print('---------测试集----------')
print_count(test)
print('---------验证集----------')
print_count(val)
print('---------数据集----------')
print_count(label)
     


