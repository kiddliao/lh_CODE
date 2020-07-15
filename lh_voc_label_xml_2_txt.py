import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import random
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train'), ('2007', 'val'),('2007', 'test')]

classes = []
# classes = ['Car','Ship','Plane','Human']
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    voc=os.path.join('..','datasets','coco_mocod','obstruct')
    in_file = open(os.path.join(voc,'Annotations',f'{image_id}.xml'))
    out_file = open(os.path.join(voc,'labels',f'{image_id}.txt'), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        cls=cls.lower().title()
        # if cls not in classes or int(difficult)==1:
        # if cls not in classes:
        #     continue
        if cls not in classes:
            classes.append(cls.title())
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()
voc = os.path.join('..', 'datasets', 'coco_infrared')

#将trainval划分为train和val
with open(os.path.join(voc, 'Main', 'trainval.txt'), 'r') as f:
    trainval = f.readlines()
    trainval = list(map(lambda x:x.strip(), trainval))
    random.shuffle(trainval)
    num = len(trainval)
    trainper, val_per = 0.9, 0.1
    train = trainval[:int(num * trainper)]
    val = trainval[int(num * trainper):]
#直接删掉没标注的
with_labeld = glob.glob(os.path.join(voc, 'Annotations', '*.xml'))
with_labeld = list(map(lambda x: x.split('/')[-1].split('.')[0], with_labeld))
train, val = set(train), set(val)
with_labeld = set(with_labeld)
train = train.intersection(with_labeld)
val = val.intersection(with_labeld)
train, val = list(train), list(val)
#把残渣给val
trainval = set(trainval)
rest = with_labeld - trainval.intersection(with_labeld)
rest = list(rest)
val += rest

with open(os.path.join(voc, 'Main', 'train.txt'), 'w') as f:
    for i in train:
        f.write(i + '\n')
with open(os.path.join(voc, 'Main', 'val.txt'), 'w') as f:
    for i in val:
        f.write(i + '\n')
            

for year, image_set in sets:
    os.makedirs(os.path.join(voc,'labels'),exist_ok=True)
    image_ids = open(os.path.join(voc, 'Main', f'{image_set}.txt')).read().strip().split()
        
    list_file = open(os.path.join(voc,f'{year}_{image_set}.txt'), 'w')
    for image_id in image_ids:
        list_file.write(os.path.join(voc, 'JPEGImages', f'{image_id}.jpg') + '\n')
        if image_set in ['train','val']:#这个数据集的test没标注
            convert_annotation(year, image_id)
    list_file.close()



# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")

