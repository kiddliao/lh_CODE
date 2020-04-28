import json
import os
from itertools import chain
#用annotation.json生成voc的trainval.txt和test.txt
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'VOC_competition'))
trainimg, valimg, testimg = [], [], []
def readjson(name):
    with open(os.path.join('Annotations',f'newinstances_{name}2017.json'), 'r') as f:
        res = []
        tmp = json.load(f)
        for t in tmp['images']:
            res.append(t['file_name'].split('.')[0])
        return res
trainimg = readjson('train')
valimg = readjson('val')
testimg = readjson('test')
with open('trainval.txt', 'w') as f:
    for i in chain(trainimg, valimg):
        f.write(i + '\n')
with open('test.txt', 'w') as f:
    for i in testimg:
        f.write(i + '\n')
        
        

        

