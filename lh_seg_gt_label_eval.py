import cv2
img = cv2.imread('/home/lh/DATASETS/road_dataset/ann_dir/train/104_mask.png')
print(img.shape)

val_map = set()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(img.shape[2]):
            val_map.add(img[i][j][k])
print(val_map)