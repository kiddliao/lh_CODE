import json
import os
from itertools import chain
#用annotation.json生成voc的trainval.txt和test.txt
# os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_FLIR', 'coco'))

trainimg, valimg, testimg = [], [], []
def readjson(name):
    with open(os.path.join('annotations',f'newinstances_{name}2017.json'), 'r') as f:
        res = []
        tmp = json.load(f)
        for t in tmp['images']:
            res.append(t['file_name'].split('.')[0])
        return res
cur_path = os.getcwd()
trainimg = readjson('train')
valimg = readjson('val')
testimg = readjson('test')
# with open('trainval.txt', 'w') as f:
#     for i in chain(trainimg, valimg):
#         f.write(i + '\n')
os.chdir('..')
cur_path = os.getcwd()
with open('train.txt', 'w') as f:
    for i in trainimg:
        f.write(os.path.join(cur_path, 'images', i + '.jpg') + '\n')
with open('test.txt', 'w') as f:
    for i in testimg:
        f.write(os.path.join(cur_path, 'images', i + '.jpg') + '\n')
with open('val.txt', 'w') as f:
    for i in valimg:
        f.write(os.path.join(cur_path, 'images', i + '.jpg') + '\n')
        
        

        

