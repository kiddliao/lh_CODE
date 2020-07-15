import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import random
from collections import defaultdict
#终极脚本 xml的voc数据集处理成darknet需要的格式
#1划分训练集测试集验证集
#2标注转换为txt
#3统计
sets = [('2007', 'train'), ('2007', 'val')]
# sets=[('2007', 'train'), ('2007', 'val'),('2007', 'test')]
# classes = ['Car','Ship','Plane','Human'] #classes有的话直接写 没的话下面统计一下
classes = [
    'knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot', 'handcuffs', 'nailpolish', 'powerbank',
    'firecrackers'
]
# classes = ['Person','Bicycle','Car']
#1划分数据集 切好了ImageSets/Main下有train.txt val.txt只想改标签不想重新划分的话这部分跳过
#1.1如果有图片没得标注文件 划分的时候需要注意删掉这些图片 用set就行

#1.2这次划分标注和图片数量一样 即都有标注 直接划分就行

# voc = os.path.join('/data', 'ssd', 'database', 'voc_infrared')
voc = ''
# voc = os.path.join('/home', 'lh', 'myhome', 'datasets', 'voc_infrared')
image_name = glob.glob(os.path.join(voc, 'JPEGImages', '*.jpg'))  #glob.glob不能识别~根目录
#切分为三个
# train_percent, val_percent, test_percent = 0.8, 0.1, 0.1
# image_num = len(image_name)
# train_num, val_num, test_num = int(train_percent * image_num), int(val_percent * image_num), int(test_percent * image_num)
# train = image_name[:train_num]
# val = image_name[train_num:train_num + val_num]
# test = image_name[train_num + val_num:]
#切分为两个
train_percent, val_percent = 0.9, 0.1
image_num = len(image_name)
train_num, val_num = int(train_percent * image_num), int(val_percent * image_num)
train = image_name[:train_num]
val = image_name[train_num:]

os.makedirs(os.path.join(voc, 'ImageSets'), exist_ok=True)
os.makedirs(os.path.join(voc, 'ImageSets', 'Main'), exist_ok=True)
os.makedirs(os.path.join(voc, 'labels'), exist_ok=True)


def gen_txt(name, data, path):
    with open(os.path.join(path, name + '.txt'), 'w') as f:
        for i in data:
            f.write(i + '\n')


gen_txt('train', list(map(lambda x: x.split('/')[-1].split('.')[0], train)), os.path.join(voc, 'ImageSets', 'Main'))
gen_txt('val', list(map(lambda x: x.split('/')[-1].split('.')[0], val)), os.path.join(voc, 'ImageSets', 'Main'))

# gen_txt('test', list(map(lambda x:x.split('/')[-1].split('.')[0],test)), os.path.join(voc,'ImageSets','Main'))
# gen_txt('trainval', list(map(lambda x: x.split('/')[-1].split('.')[0], train + val)), os.path.join(voc, 'ImageSets', 'Main'))

# gen_txt('2007_train', train, voc)
# gen_txt('2007_val', val, voc)
# gen_txt('2007_test', test, voc)


#2标注转换txt
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id, voc):
    in_file = open(os.path.join(voc, 'Annotations', f'{image_id}.xml'))
    out_file = open(os.path.join(voc, 'labels', f'{image_id}.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # cls = cls.title()  #首字母大写
        # if cls not in classes or int(difficult)==1:
        if cls not in classes:
            continue
        # if cls not in classes:
        #     classes.append(cls.title())
        stat[cls] += 1
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


for year, image_set in sets:
    image_ids = open(os.path.join(voc, 'ImageSets', 'Main', f'{image_set}.txt'), 'r').read().strip().split()
    list_file = open(os.path.join(voc, f'{year}_{image_set}.txt'), 'w')
    stat = defaultdict(int)
    for image_id in image_ids:
        list_file.write(os.path.join(voc, 'JPEGImages', f'{image_id}.jpg') + '\n')
        if image_set in ['train', 'val', 'test']:  #如果测试集没标注的话这里就不加测试集
            convert_annotation(year, image_id, voc)
    list_file.close()
    #3统计
    stat = sorted(stat.items(), key=lambda x: x[0])
    print(f'{image_set}集的数据分布为{stat}\n')
print(f'类别数一共{len(classes)},分别是{classes}')