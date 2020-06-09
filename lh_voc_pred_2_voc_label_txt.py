import os
import json
import glob
from collections import defaultdict
width, height = 1280, 720
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_mocod', 'test'))
with open('trans_result.txt', 'r') as f:
    result = f.readlines()

dic_class = {
    'Car': 0,
    'Ship': 1,
    'Plane': 2,
    'Human': 3
}
result = list(map(lambda x: x.strip(), result))
ans = defaultdict(list)
for i in result:
    img_name, conf, xmin, ymin, xmax, ymax = i.split(' ')
    cid = dic_class[img_name.split('_')[0].title()]
    img_name = img_name.lower() + '.txt'
    conf, xmin, ymin, xmax, ymax = list(map(float, [conf, xmin, ymin, xmax, ymax]))
    xmin /= width
    xmax /= width
    ymin /= height
    ymax /= height
    w, h = xmax - xmin, ymax - ymin
    xc, yc = xmin + w / 2, ymin + h / 2
    ans[img_name].append([cid, xc, yc, w, h])
for k, v in ans.items():
    with open(k, 'w') as f:
        for i in v:
            i = list(map(str, i))
            tmp = ' '.join(i)
            f.write(tmp + '\n')


