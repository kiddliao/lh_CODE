import cv2
import json
import glob
import os

NAME = 'val'

ifile = glob.glob(os.path.join(f'{NAME}_mmdetection', '*.json'))
for i in range(len(ifile)):
    img_name = ifile[i].split('\\')[-1].split('.')[0]
    with open(ifile[i], 'r') as f:
        context = json.load(f)
    new_context = []
    print(f'当前进度{i+1}/{len(ifile)}......')
    print(f'当前图片名字是{img_name}.png')
    print('-'*30+f'筛选前有{len(context)}个框'+'-'*30)
    for j in range(len(context)):
        img = cv2.imread(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray', NAME, img_name + '.png'))
        category_name, bbox = list(context[j].items())[0]
        if category_name not in ["Nodule", "Emphysema", "Fracture"]:
            new_context.append(context[j])
            continue
        print(f'当前框的信息是{bbox}')
        print(f'置信度是{bbox[-1]}')
        x1, y1, x2, y2,score = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, '{}, {:.3f}'.format(category_name, 1),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (255, 255, 0), 2)
        cv2.namedWindow(img_name, 0)
        cv2.resizeWindow(img_name, 600, 600)
        cv2.imshow(img_name, img)
        if cv2.waitKey(0) & 0xFF == ord('y'):
            print('这个框-----删掉!!!')
            continue
        new_context.append(context[j])
        print('这个框没问题')
    cv2.destroyAllWindows()
    with open(ifile[i], 'w') as f:
        json.dump(new_context, f)
        print('-'*30+f'筛选后还有{len(new_context)}个框'+'-'*30)
        print(img_name+'.png筛选成功,next')





        
    