from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import FlirDataset, collater, Resizer, Normalizer, Augmenter
from tqdm.autonotebook import tqdm
import cv2
import numpy as np
category={0:'person',1:'bicycle',2:'car'}

root_dir = 'coco'
set_name = 'train2017'
mspath='mean_std.txt'
batch_size = 1
#先求一下均值和方差
training_set = FlirDataset(root_dir, set_name, mean_std_path=mspath
, cal_mean_std=False, transform=transforms.Compose([
    Normalizer(mean_std_path=mspath),Augmenter(),Resizer(512)]))
# a = training_set[0]  #取一个样本看看
# cv2.imshow('image', a['img'].numpy())
# cv2.waitKey(0)
# cv2.destroyAllWindows()
training_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'drop_last': True,
                    'collate_fn': collater,
                    'num_workers': 0}
training_generator = DataLoader(training_set, **training_params)
progress_bar=tqdm(training_generator)
for iter, data in enumerate(progress_bar):
    imgs = data['img']
    annot = data['annot']
    print(imgs.shape,annot.shape)
    print(imgs.size(), annot.size())
    # pic = imgs.squeeze().numpy()
    # # pic=pic.reshape((*pic.shape,1)) #单通道图
    # for box in annot.squeeze():
    #     print(box.size())
    #     x1, y1, x2, y2, id = box.squeeze()
    #     x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
    #     cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)  #2是线的宽度
    #     cv2.putText(pic, '{}, {:.3f}'.format(category[int(id)], 1),
    #                 (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                 (255, 255, 0), 1)
    # cv2.namedWindow('img')
    # cv2.imshow('img', pic)
    # cv2.waitKey(0)
    pass
# cv2.destroyAllWindows()
