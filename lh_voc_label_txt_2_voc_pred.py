import os
import json
import glob
from math import ceil
from collections import defaultdict
width, height = 1280, 720
path = os.path.join('..', 'REMOTE', 'datasets', 'coco_mocod', 'test')

dic_class = {
    0: 'Car',
    1: 'Ship',
    2: 'Plane',
    3: 'Human'
}

txt = glob.glob(os.path.join(path, '*_????.txt'))
print(len(txt))

print(len(txt))
with open(os.path.join('xray_pred', 'new_trans_result.txt'), 'w') as f:
    for i in txt:
        with open(i, 'r') as f1:
            img_name=i.split('\\')[-1].split('.')[0].title()
            single_label = f1.readlines()
            for j in single_label:
                cid, xc, yc, w, h = list(map(float, j.split(' ')))
                xc, yc, w, h = xc * width, yc * height, w * width, h * height
                xmin, ymin, xmax, ymax = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                tmp = [img_name, 1, xmin, ymin, xmax, ymax]
                tmp = list(map(str, tmp))
                tmp = ' '.join(tmp)
                f.write(tmp+'\n')
                
#精准匹配 乘除width和height对框的变化不大
# with open(os.path.join('xray_pred', 'trans_result.txt'), 'r') as f:
#     old = f.readlines()
#     old = list(map(lambda x:x.strip(), old))
# with open(os.path.join('xray_pred', 'new_trans_result.txt'), 'r') as f:
#     new = f.readlines()
#     new = list(map(lambda x: x.strip(), new))
# old_tmp = list(map(lambda x: x.split(' '), old))
# temp = list(map(lambda x: (x[0], x[1], *list(map(lambda x: round(x, 0), list(map(float, x[2:]))))), old_tmp))
# conf_map = {}
# for i in range(len(temp)):
#     conf_map[(temp[i][0], temp[i][2], temp[i][3], temp[i][4], temp[i][5])] = [temp[i][1],*old[i].split(' ')]

# old_tmp = list(map(lambda x: (x[0], *list(map(lambda x:round(x,0), list(map(float, x[2:]))))), old_tmp))
# new_tmp = list(map(lambda x: x.split(' '),new))
# new_tmp = list(map(lambda x: (x[0], *list(map(lambda x:round(x,0), list(map(float, x[2:]))))), new_tmp))
# old_tmp = set(old_tmp)
# new_tmp = set(new_tmp)
# final = old_tmp.intersection(new_tmp)
# print(old_tmp - final)
# print(new_tmp - final)
# final = list(final)

# with open(os.path.join('xray_pred','final_result.txt'), 'w') as f:
#     for i in sorted(final, key=lambda x: x[0]):
#         key = tuple(i)
#         conf, img_name, old_conf, xmin, ymin, xmax, ymax = conf_map[key]
#         img_name = img_name.title()
#         tmp = [img_name, old_conf, xmin, ymin, xmax, ymax]
#         tmp = ' '.join(tmp)
#         f.write(tmp + '\n')
        


#范围匹配 原来的框和现在的框的差别不超过2就是一个框
def contrast(a, b):
    if abs(a[0] - b[0]) < 2 and abs(a[1] - b[1]) < 2 and abs(a[2] - b[2]) < 2 and abs(a[3] - b[3]) < 2:
        return True
    return False

with open(os.path.join('xray_pred', 'trans_result.txt'), 'r') as f:
    old = f.readlines()
    old = list(map(lambda x:x.strip(), old))
with open(os.path.join('xray_pred', 'new_trans_result_copy.txt'), 'r') as f:
    new = f.readlines()
    new = list(map(lambda x: x.strip(), new))
old_tmp = list(map(lambda x: x.split(' '), old))
temp = list(map(lambda x: (x[0], x[1], *list(map(lambda x: round(x, 0), list(map(float, x[2:]))))), old_tmp))
conf_map = defaultdict(list)
for i in range(len(temp)):
    conf_map[temp[i][0]].append(list(map(float,old[i].split(' ')[1:])))

new_tmp = list(map(lambda x: x.split(' '),new))
new_tmp = list(map(lambda x: [x[0], *list(map(float, x[2:]))], new_tmp))
with open(os.path.join('xray_pred', 'final_result.txt'), 'w') as f:
    for i in new_tmp:
        img_name, xmin, ymin, xmax, ymax = i
        for j in conf_map[img_name]:
            if contrast(i[1:], j[1:]):
                conf = j[0]
                break
        tmp = [img_name, conf, xmin, ymin, xmax, ymax]
        tmp = list(map(str, tmp))
        tmp = ' '.join(tmp)
        f.write(tmp + '\n')
        

    
