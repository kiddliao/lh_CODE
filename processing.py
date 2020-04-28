import cv2
import os
import random
import json
#随即打印图片以及gt框来检验数据集标注是否有问题
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_FLIR', 'coco'))
category={1:'person',2:'bicycle',3:'car',4:'dog'}
# category={1:'car',2:'ship',3:'human',4:'plane'}
with open(os.path.join('annotations', 'newinstances_train2017.json'), 'r') as f:
    label = json.load(f)
    images = label['images']
    categories = label['categories']
    annotations = label['annotations']
    random.shuffle(images)
    for img in images[:20]:
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
        cv2.imshow(img_name, pic)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

        
