import cv2
# img = cv2.imread(
#     '/home/lh/DATASETS/VOC2012_AUG/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png', cv2.IMREAD_COLOR)
img = cv2.imread(
    '/home/lh/DATASETS/road_dataset/ann_L_dir/train/871780_sat.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)

val_map = {}
# gt_val = [[0]*img.shape[1] for i in range(img.shape[0])]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        val_map[int(img[i][j])] = val_map.setdefault(int(img[i][j]), 0) + 1
        # val_map.add(tuple(img[i][j].tolist()))
        # gt_val[i][j] = img[i][j][0]
print(val_map)
# for i in gt_val:
#     print(i)
