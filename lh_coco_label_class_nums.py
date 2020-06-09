import os
import json
from collections import defaultdict
#读取coco的标注显示出每类的样本个数
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray', 'annotations'))

def count_nums(paths):
    categories = []
    annotations = []
    category_map = defaultdict(list)
    for i in paths:
        with open(f'newinstances_{i}2017.json', 'r') as f:
            tmp = json.load(f)
            annotations += tmp['annotations']
            categories = tmp['categories']
    pass
    for i in categories:
        category_map[i['id']] += [i['name'], 0]
    for i in annotations:
        if i['category_id'] not in list(range(1, 1 + len(categories))):
            continue
        category_map[i['category_id']][1] += 1
    if len(paths) == 1:
        name = paths[0]
    elif len(paths) == 2:
        name = '_'.join(paths)
    else:
        name = 'all'
    print('-'*80)
    print(f'{name}的每类样本分布')
    for i,v in sorted(category_map.items(), key=lambda a: (a[0], a[1])):  #a={a[0],a[1]} 代表迭代中的一个键值对 优先按值排序则把a[1]放前面
        print(f'{v[0]}的样本数量为{v[1]}')
    print('-' * 80)
    print('\n\n')
    
    









count_nums(['train'])
count_nums(['val'])
count_nums(['train','val'])