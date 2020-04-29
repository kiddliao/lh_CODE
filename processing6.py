import random
import json
import os
import re
import glob
#把coco格式数据集转换为voc格式(txt)
# rg = re.compile('\d{5}') #将图片名格式化为000001 ...
#转换val的标注文件
os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_shape'))
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)
os.makedirs('test', exist_ok=True)



def create_annot_txt(name):
    with open(os.path.join('annotations',f'instances_{name}.json'), 'r+',encoding='utf-8') as f:
    # with open(os.path.join('..','datasets','FLIR_ADAS_1_3','val','thermal_annotations.json'), 'r+',encoding='utf-8') as f:
        res = json.load(f)
    class_id = res["categories"]
    annotations = res["annotations"]
    images = res["images"]
    #建立image_id和file_name的映射
    dic = {}
    for i in images:
        dic[i['id']]=i['file_name']

    #coco格式的bbox是(x,y,w,h)框左上角坐标和框宽高 转换成中心点坐标和框宽高

    # init_id = int(rg.search(images[0]['file_name']).group())
    for i in range(len(annotations)):
        file_name=dic[annotations[i]['image_id']]
        # cur_id = int(annotations[i]['image_id'])
        # real_id = str(init_id + cur_id).rjust(5, '0')
        label_id = annotations[i]['category_id']
        if label_id == 17: label_id = 4
        annotations[i]['bbox'][0]+=annotations[i]['bbox'][2]/2
        annotations[i]['bbox'][1]+=annotations[i]['bbox'][3]/2
        writein_label = ' '.join(list(map(str,[label_id-1] + annotations[i]['bbox']))) #录入的时候id-1
        
        with open(os.path.join('labels','{}.txt'.format(file_name.split('.')[0])), 'a+') as f:
            f.write(writein_label+'\n')
    
create_annot_txt('train')
create_annot_txt('val')
# create_annot_txt('test')


# #对无样本图片新建空标签文本 但是这个项目不接受空的label所以注释掉
# img_name = os.listdir(os.path.join('..','datasets','FLIR_pt_yolov3','images'))
# # label_name = os.listdir(os.path.join('..', 'datasets', 'FLIR_pt_yolov3', 'labels'))
# img_name = set(list(map(lambda x: x[:5], img_name)))
# # label_name = set(list(map(lambda x: x[:5], label_name)))
# # no_name=list(img_name - label_name)
# # for i in range(len(no_name)):
# #     with open(os.path.join('..','datasets','FLIR_pt_yolov3','labels','{}.txt'.format(no_name[i])), 'a+') as f:
# #         print(no_name[i])

img_name = glob.glob(os.path.join('*', '*.jpg'))
del_img = glob.glob(os.path.join('images', '*.jpg'))
img_name=list(set(img_name)-set(del_img))
#分配train val test
img_name = list(img_name)
trainpercent, valpercent, testpercent = 0.8, 0.1, 0.1
n = len(img_name)
trainnum, valnum = int(trainpercent * n), int(valpercent * n)
testnum = n - trainnum - valnum
random.seed(0)
random.shuffle(img_name)
train = img_name[:trainnum]
val = img_name[trainnum:trainnum + valnum]
test = img_name[trainnum + valnum:]

cur_path = os.getcwd()
with open('train.txt','w') as f:
    for i in range(len(train)):
        f.write(os.path.join(cur_path, 'images', train[i]) + '\n')
with open('test.txt','w') as f:
    for i in range(len(test)):
        f.write(os.path.join(cur_path, 'images', test[i]) + '\n')
with open('valid.txt','w') as f:
    for i in range(len(val)):
        f.write(os.path.join(cur_path, 'images', val[i]) + '\n')

#train的图片路径和label来生成anchor
f1 = open('train.txt', 'r')
train_path = f1.readlines()
f1.close()
with open('train_all_labels.txt', 'w') as f:
    for i in range(len(train_path)):
        image_path = train_path[i].strip()
        # image_name=image_path.split('/')[-1].split('.')[0]
        image_name=image_path.split('\\')[-1].split('.')[0]
        # image_id = rg.search(image_path).group()
        with open(os.path.join('labels', image_name + '.txt'), 'r') as f2:
            labels = f2.readlines()
            image_label = ''
            for j in range(len(labels)):
                cur_label = labels[j].strip().split()
                cur_label = cur_label[1:] + [cur_label[0]]
                image_label += ','.join(cur_label)+' '
        write_in = image_path + ' ' + image_label.strip()
        f.write(write_in+'\n')
os.system('cp test/*.jpg images')           
os.system('cp train/*.jpg images')           
os.system('cp val/*.jpg images')           
    