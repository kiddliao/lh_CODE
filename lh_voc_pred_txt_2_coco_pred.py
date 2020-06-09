import json
import os
import glob
dict_class = {
    'Car': 1,
    'Truck': 2,
    'Van': 3
}
dir_name = 'results_test_multi_89_59'
NAME = 'test'

with open(os.path.join('xray_pred2', f'newinstances_{NAME}2017.json'),'r') as f:
    gt = json.load(f)
img_map = {}
for i in gt['images']:
    img_map[i['file_name'].split('.')[0]] = i['id']

coco_sample = {"image_id": 0, "bbox": [], "score": 0, "category_id": 0}
pred_txt = glob.glob(os.path.join(dir_name, '*.txt'))
pred_txt = list(map(lambda x: x.strip().split('/')[-1], pred_txt))
final = []
for i in pred_txt: 
    with open(os.path.join(dir_name, i), 'r') as f:
        cid=dict_class[i.split('.')[0]]
        class_txt = f.readlines()
        class_txt = list(map(lambda x: x.strip().split(' '), class_txt))
        class_txt = list(map(lambda x: [x[0], float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])],class_txt))
        for j in class_txt:
            img_name, conf, xmin, ymin, xmax, ymax = j
            img_id = img_map[img_name]
            coco_tmp = coco_sample.copy()
            coco_tmp['image_id'] = img_id
            coco_tmp['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            coco_tmp['score'] = conf
            coco_tmp['category_id'] = cid
            final.append(coco_tmp.copy())
with open(os.path.join('xray_pred2', 'mocod_test1.json'), 'w') as f:
    json.dump(final, f)


            
            


        
    

        