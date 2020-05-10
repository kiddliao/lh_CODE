import cv2
import os
import random
import json
#随机打印coco数据集图片以及gt框来检验数据集标注是否有问题
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_flir','coco'))
category={1:'person',2:'bicycle',3:'car',4:'dog'}
# category={1:'car',2:'ship',3:'human',4:'plane'}
# category={1:"Consolidation",
# 2:"Fibrosis",
# 3:"Effusion",
# 4:"Nodule",
# 5:"Mass",
# 6:"Emphysema",
# 7:"Calcification",
# 8:"Atelectasis",
# 9:"Fracture"
# }
with open(os.path.join('annotations','newinstances_train2017.json'), 'r') as f:
    label = json.load(f)
    images = label['images']
    categories = label['categories']
    annotations = label['annotations']
    random.shuffle(images)
    for img in images[:15]:
        bbox = []
        cid = 0
        img_name = img['file_name']
        img_id = img['id']
        flag=0
        for res in annotations:#这两个if先后顺序很重要 倒换的话第二个if得变成elif
            if res['image_id'] != img_id and flag>0:break
            if res['image_id'] == img_id:
                flag += 1
                bbox.append([*res['bbox'], res['category_id']])
        pic = cv2.imread(os.path.join('train2017', img_name))
        for box in bbox:
            x, y, w, h, id = box
            cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)  #2是线的宽度
            cv2.putText(pic, '{}, {:.3f}'.format(category[id], 1),
                        (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        cv2.namedWindow(img_name)
        # cv2.resizeWindow(img_name, 416, 416)
        cv2.imshow(img_name, pic)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#x-ray比赛的数据
# with open(os.path.join("train_x-ray.json"), 'r') as f:
#     label = json.load(f)
#     for i in range(400):
#         if label[i]['boxes'] == [] or 'Fracture' not in label[i]['syms']:
#             continue
#         else:
#             img_name = label[i]['file_name']
#             pic = cv2.imread(os.path.join('train', img_name), 0)
#             for j in range(len(label[i]["syms"])):
#                 x1, y1, x2, y2 = label[i]['boxes'][j]
#                 cv2.rectangle(pic, (x1, y1), (x2 , y2), (0, 255, 0), 2)  #2是线的宽度
#                 cv2.putText(pic, '{}, {:.3f}'.format(label[i]['syms'][j], 1),
#                             (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
#                             (255, 255, 0), 2)
#             cv2.namedWindow(img_name, 0)
#             cv2.resizeWindow(img_name, 416, 416)
#             # pic = cv2.resize(pic, 416, 416)
#             cv2.imshow(img_name, pic)
#             cv2.waitKey(0)
# cv2.destroyAllWindows()


    
