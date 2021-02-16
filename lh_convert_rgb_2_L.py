from PIL import Image
import os
import glob
import numpy as np
import torch

class_names = ['background', 'road']
name_platte = {'background': 0, 'road': 255}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
def convert(dir_name):
    paths = glob.glob(os.path.join('ann_dir', dir_name, '*.png'))
    for i in range(len(paths)):
        img = Image.open(paths[i])
        img_L = img.convert('L')
        img_L = np.array(img_L)
        img_L = torch.from_numpy(img_L).to(device)

        for label_id, label_name in enumerate(class_names):
            palette_id = name_platte[label_name]
            tmp_matrix = img_L == palette_id
            img_L[tmp_matrix] = label_id
        img_L = np.array(img_L.to(device_cpu))
        img_L = Image.fromarray(img_L)
        img_L.save(paths[i].replace('ann_dir', 'ann_L_dir'))
        print('{} to {} 成功转换 {}/{}'.format(img.mode, img_L.mode, i, len(paths)))
        
convert('train')
convert('val')
