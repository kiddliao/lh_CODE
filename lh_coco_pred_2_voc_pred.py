import random
import json
import os
import re
import glob
import shutil
import cv2
from itertools import chain
from collections import defaultdict
#xray比赛将coco格式预测结果转换voc类似格式
dict_class = {
        '1':'Consolidation',
        '2':'Fibrosis',
        '3':'Effusion',
        '4':'Nodule',
        '5':'Mass',
        '6':'Emphysema',
        '7':'Calcification',
        '8':'Atelectasis',
        '9':'Fracture'
    }
with open(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray', 'annotations', 'newinstances_test2017.json'), 'r') as f:
    empty_test = json.load(f)
img_map = {}
for i in range(len(empty_test['images'])):
    img_map[empty_test['images'][i]['id']] = empty_test['images'][i]['file_name']

def parse_annot_txt(name):
    with open(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray', 'annotations', 'newinstances_{}2017.json'.format(name)), 'r') as f:
        empty_test = json.load(f)
    img_map = {}
    for i in range(len(empty_test['images'])):
        img_map[empty_test['images'][i]['id']] = empty_test['images'][i]['file_name']
    all_res=defaultdict(list)
    os.makedirs(name+'_mmdetection',exist_ok=True)
    with open(os.path.join(f'new_{name}.json'), 'r+',encoding='utf-8') as f:
    # with open(os.path.join('..','datasets','FLIR_ADAS_1_3','val','thermal_annotations.json'), 'r+',encoding='utf-8') as f:
        res = json.load(f)
    #coco格式的bbox是(x,y,w,h)框左上角坐标和框宽高 转换成比赛要求的左上角坐标和右下角坐标
    # init_id = int(rg.search(images[0]['file_name']).group())
    for i in range(len(res)):
        bbox = res[i]['bbox']
        cid = str(res[i]['category_id'])
        img_id = res[i]['image_id']
        confi = res[i]['score']
        file_name = img_map[img_id]
        # pic = cv2.imread(os.path.join('JPEGImages',file_name))
        # pic_shape=list(pic.shape)
        # cur_id = int(annotations[i]['image_id'])
        # real_id = str(init_id + cur_id).rjust(5, '0')
        # label_id = annotations[i]['category_id']
        # if label_id == 17: label_id = 4
        # annotations[i]['bbox'][0] += annotations[i]['bbox'][2] / 2
        # annotations[i]['bbox'][1] += annotations[i]['bbox'][3] / 2
        # annotations[i]['bbox'][0] /= pic_shape[1]
        # annotations[i]['bbox'][2] /= pic_shape[1]
        # annotations[i]['bbox'][1] /= pic_shape[0]
        # annotations[i]['bbox'][3] /= pic_shape[0]
        bbox[2]+=bbox[0]
        bbox[3]+=bbox[1]

        cur={dict_class[cid]:bbox+[confi]}
        # writein_label = ' '.join(list(map(str,[label_id-1] + annotations[i]['bbox']))) #录入的时候id-1
        all_res[file_name].append(cur.copy())
        # with open(os.path.join(name+'mmdetection','{}.json'.format(file_name.split('.')[0])), 'a+') as f:
        #     f.write(writein_label+'\n')
    for k, v in all_res.items():
        with open(os.path.join(name + '_mmdetection', '{}.json'.format(k.split('.')[0])), 'w') as f:
            json.dump(v, f)


# parse_annot_txt('test')
parse_annot_txt('final')
