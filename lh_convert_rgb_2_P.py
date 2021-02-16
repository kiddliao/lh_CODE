from PIL import Image
import os
import glob

def convert(dir_name):
    paths = glob.glob(os.path.join('ann_dir', dir_name, '*.png'))
    for i in range(len(paths)):
        img = Image.open(paths[i])
        img_P = img.convert('P')
        img_P.save(paths[i].replace('ann_dir', 'ann_P_dir'))
        print('{} to {} 成功转换 {}/{}'.format(img.mode, img_P.mode, i, len(paths)))
        
convert('train')
convert('val')