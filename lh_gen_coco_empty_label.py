import glob
import json
import os
#mmdetection生成测试集需要的空标注文件
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray'))
with open(os.path.join('annotations','newinstances_train2017.json'),'r') as f:
    train = json.load(f)
test = {}
test['categories'] = train['categories']
test['info'] = train['info']
test['licenses'] = train['licenses']
test['images'] = []
img_tmp = train['images'][0]
test_img_name = glob.glob(os.path.join('final', '*.png'))
for i in range(len(test_img_name)):
    file_name = test_img_name[i].strip().split('\\')[-1].split('.')[0]+'.jpg'
    img_tmp['file_name'] = file_name
    img_tmp['id'] = i
    test['images'].append(img_tmp.copy())
with open(os.path.join('annotations', 'newinstances_final2017.json'), 'w') as f:
    json.dump(test, f)


