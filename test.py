import os
import json
os.chdir(os.path.join('..','REMOTE','datasets','coco_flir','coco','annotations'))
with open('newinstances_val2017.json','r') as f:
    res = json.load(f)
    print(len(res))
pass






# import os
# import glob
# os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_mocod'))
# with open(os.path.join('obstruct', 'Main', 'test.txt')) as f:
#     test = f.readlines()
# test = list(map(lambda x: x.strip(), test))
# test = set(test)
# jpg = glob.glob(os.path.join('test', '*.jpg'))
# jpg = list(map(lambda x: x.strip().split('\\')[-1].split('.')[0], jpg))
# jpg = set(jpg)
# print(jpg-test)