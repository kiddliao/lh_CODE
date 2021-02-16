import glob
import os
import cv2
import json
import torch
import numpy as np

paths = glob.glob(os.path.join('*', '*', '*.jpg'))
print(len(paths))

def get_mean_std(paths):
    means = torch.Tensor([0, 0, 0])
    stdevs = torch.Tensor([0, 0, 0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    means = means.to(device)
    stdevs = stdevs.to(device)
    print(device)
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32) / 255.
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)  #hwc 2 chw
        img = torch.from_numpy(img)
        img = img.to(device)


        for j in range(3):
            means[j] += img[j, :, :].mean()
            stdevs[j] += img[j, :, :].std()

    means = means.tolist()
    stdevs = stdevs.tolist()
    means = np.asarray(means) / len(paths)
    stdevs = np.asarray(stdevs) / len(paths)
    print("{} : normMean = {}".format(type, means))
    print("{} : normstdevs = {}".format(type, stdevs))
    
    with open('mean_std.txt', 'w') as f:
        json.dump({'means': list(means), 'stdevs': list(stdevs)}, f)

get_mean_std(paths)