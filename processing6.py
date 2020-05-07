import random
import json
import os
import re
import glob
import shutil
import cv2
from itertools import chain
#把coco格式数据集转换为voc格式(txt)
# rg = re.compile('\d{5}') #将图片名格式化为000001 ...
#转换val的标注文件
# os.chdir(os.path.join('..', 'datasets', 'coco_xray'))
os.makedirs('JPEGImages', exist_ok=True)
os.makedirs('Annotations', exist_ok=True)
# os.makedirs('test', exist_ok=True)
# os.makedirs(os.path.join('labels','test'), exist_ok=True)
# os.makedirs(os.path.join('labels','train'), exist_ok=True)
# os.makedirs(os.path.join('labels','val'), exist_ok=True)



def create_annot_txt(name):
    with open(os.path.join('coco','annotations',f'newinstances_{name}2017.json'), 'r+',encoding='utf-8') as f:
    # with open(os.path.join('..','datasets','FLIR_ADAS_1_3','val','thermal_annotations.json'), 'r+',encoding='utf-8') as f:
        res = json.load(f)
    class_id = res["categories"]
    annotations = res["annotations"]
    images = res["images"]
    #建立image_id和file_name的映射
    dic = {}
    for i in images:
        dic[i['id']]=i['file_name']

    #coco格式的bbox是(x,y,w,h)框左上角坐标和框宽高 转换成中心点坐标和框宽高

    # init_id = int(rg.search(images[0]['file_name']).group())
    for i in range(len(annotations)):
        file_name=dic[annotations[i]['image_id']]
        pic = cv2.imread(os.path.join('JPEGImages',file_name))
        pic_shape=list(pic.shape)
        # cur_id = int(annotations[i]['image_id'])
        # real_id = str(init_id + cur_id).rjust(5, '0')
        label_id = annotations[i]['category_id']
        # if label_id == 17: label_id = 4
        annotations[i]['bbox'][0] += annotations[i]['bbox'][2] / 2
        annotations[i]['bbox'][1] += annotations[i]['bbox'][3] / 2
        annotations[i]['bbox'][0] /= pic_shape[0]
        annotations[i]['bbox'][2] /= pic_shape[0]
        annotations[i]['bbox'][1] /= pic_shape[1]
        annotations[i]['bbox'][3] /= pic_shape[1]

        
        writein_label = ' '.join(list(map(str,[label_id-1] + annotations[i]['bbox']))) #录入的时候id-1
        
        with open(os.path.join('Annotations','{}.txt'.format(file_name.split('.')[0])), 'a+') as f:
            f.write(writein_label+'\n')
    
create_annot_txt('train')
create_annot_txt('val')
# create_annot_txt('test')


#对无样本图片新建空标签文本 但是这个项目不接受空的label所以注释掉
img_name = os.listdir('JPEGImages')
label_name = os.listdir('Annotations')
img_name = set(list(map(lambda x:x.split('.')[0], img_name)))
label_name = set(list(map(lambda x:x.split('.')[0], label_name)))
no_name=list(img_name - label_name)
for i in range(len(no_name)):
    with open(os.path.join('Annotations','{}.txt'.format(no_name[i])), 'a+') as f:
        print(no_name[i])

# img_name = glob.glob(os.path.join('*', '*.jpg'))
# del_img = glob.glob(os.path.join('images', '*.jpg'))
# img_name=list(set(img_name)-set(del_img))
# #分配train val test
# img_name = list(img_name)
# trainpercent, valpercent, testpercent = 0.8, 0.1, 0.1
# n = len(img_name)
# trainnum, valnum = int(trainpercent * n), int(valpercent * n)
# testnum = n - trainnum - valnum
# random.seed(0)
# random.shuffle(img_name)
# train = img_name[:trainnum]
# val = img_name[trainnum:trainnum + valnum]
# test = img_name[trainnum + valnum:]

cur_path = os.getcwd()
# with open('train.txt','w') as f:
#     for i in range(len(train)):
#         f.write(os.path.join(cur_path, 'images', train[i]) + '\n')
# with open('test.txt','w') as f:
#     for i in range(len(test)):
#         f.write(os.path.join(cur_path, 'images', test[i]) + '\n')
# with open('valid.txt','w') as f:
#     for i in range(len(val)):
#         f.write(os.path.join(cur_path, 'images', val[i]) + '\n')

#从json读取数据集分配
with open(os.path.join('coco','annotations', 'newinstances_train2017.json'), 'r+', encoding='utf-8') as f:
    train = json.load(f)
with open(os.path.join('coco','annotations', 'newinstances_val2017.json'), 'r+', encoding='utf-8') as f:
    val = json.load(f)
with open('train.txt','w') as f:
    for i in range(len(train['images'])):
        f.write(os.path.join(cur_path, 'JPEGImages', train['images'][i]['file_name']) + '\n')
with open('valid.txt','w') as f:
    for i in range(len(val['images'])):
        f.write(os.path.join(cur_path, 'JPEGImages', val['images'][i]['file_name']) + '\n')
with open('trainval.txt','w') as f:
    for i in chain(train['images'],val['images']):
        f.write(os.path.join(cur_path, 'JPEGImages',i['file_name']) + '\n')
    


#train的图片路径和label来生成kmeans的anchor
f1 = open('train.txt', 'r')
train_path = f1.readlines()
f1.close()
with open('train_all_labels.txt', 'w') as f:
    for i in range(len(train_path)):
        image_path = train_path[i].strip()
        image_name=image_path.split('/')[-1].split('.')[0]
        # image_name=image_path.split('\\')[-1].split('.')[0]
        # image_id = rg.search(image_path).group()
        with open(os.path.join('Annotations', image_name + '.txt'), 'r') as f2:
            labels = f2.readlines()
            image_label = ''
            for j in range(len(labels)):
                cur_label = labels[j].strip().split()
                cur_label = cur_label[1:] + [cur_label[0]]
                image_label += ','.join(cur_label)+' '
        write_in = image_path + ' ' + image_label.strip()
        f.write(write_in + '\n')
        
#把labels下的txt放入labels/train val test
# def move(name, res):
#     for i in res:
#         txtpath = i.strip().split('/')[-1].split('.')[0]+'.txt'
#         # txtpath = i.strip().split('\\')[-1].split('.')[0]+'.txt'
#         shutil.move(os.path.join('labels', txtpath), os.path.join('labels', name, txtpath))

# move('train', train)
# move('test', test)
# move('val', val)


# # os.system('cp test/*.jpg images')           
# os.system('cp train/*.jpg images')           
# os.system('cp val/*.jpg images')           
    