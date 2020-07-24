import cv2
import os
import random
import json
import glob
from collections import defaultdict
#随机打印voc数据集图片以及gt框来检验数据集标注是否有问题
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_shape'))
pic_paths = glob.glob(os.path.join('JPEGImages','*.jpg'))
pic = cv2.imread(pic_paths[0])
pic_shape=list(pic.shape)
stat=defaultdict(int)
# category={
#     1: 'knife',
#     2: 'scissors',
#     3: 'lighter',
#     4: 'zippooil',
#     5: 'pressure',
#     6: 'slingshot',
#     7: 'handcuffs',
#     8: 'nailpolish',
#     9: 'powerbank',
#     10: 'firecrackers'
# }
category = {1: 'rectangle', 2: 'circle'}

label_paths = glob.glob(os.path.join('labels', '*.txt'))#直接读取每张图的标签
# with open(os.path.join('train_all_labels.txt'), 'r') as f:#读取train_all_labels.txt的标签
    # label_paths = f.readlines()
label_paths = list(map(lambda x: x.strip().split(' ')[0], label_paths))
random.shuffle(label_paths)
for i in range(len(label_paths)):
    #直接读标签
    label_path = label_paths[i]
    img_name = label_path.strip().split('\\')[-1].split('.')[0]
    pic=cv2.imread(os.path.join('JPEGImages', img_name + '.jpg'))
    _h, _w, _c=pic.shape
    with open(label_path, 'r') as f:
        res = f.readlines()

    # 读train_all_labels.txt
    # img_path = label_paths[i]
    # img_name = img_path.split('\\')[-1]
    # label_path=os.path.join('labels', img_name.split('.')[0] + '.txt')
    # pic=cv2.imread(os.path.join('images', img_name))
    # with open(label_path, 'r') as f:
    #     res = f.readlines()

    for i in res:
        cid = int(i.strip().split(' ')[0])
        cname = category[cid + 1]
        stat[cname]+=1
        xc, yc, w, h = list(map(float, i.strip().split(' ')[1:]))
        # x, y, w, h = int(xc - w / 2), int(yc - h / 2), int(w), int(h)
        x, y, w, h = int((xc - w / 2) * _w), int((yc - h / 2) * _h), int(w * _w), int(h * _h)

        cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)  #2是线的宽度
        cv2.putText(pic, '{}, {:.3f}'.format(cname, 1),
                    (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
    cv2.namedWindow(img_name)
    cv2.imshow(img_name, pic)
    cv2.waitKey(0)
cv2.destroyAllWindows()
# print(stat)





