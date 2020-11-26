from __future__ import print_function
import logging as log
import json
import os
import cv2
import numpy as np

import argparse
import os
import warnings
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

log.basicConfig(level=log.DEBUG)
os.chdir(os.path.join("/usr","local",'ev_sdk',"yolov5"))

class_name = {
    1: "vendors"
}
img_size=416

def init():
    weights='/usr/local/ev_sdk/model/best.pth'
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    return model
    
    


def process_image(model, img0, args=None):
    imgsz = check_img_size(img_size, s=model.stride.max())

    img = letterbox(img0, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    device = select_device('')
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, merge=False, classes=None, agnostic=False) #此处可选merge nms


    # result = inference_detector(model, img)
    detect_objs = []
    #单分类 只有一个vendors
    for box in pred[0]:
        xmin, ymin, xmax, ymax, conf, cid = box
        cid = int(cid)
        cname = class_name[cid]
        if min(xmin, ymin, xmax, ymax) > 1.0:
            detect_objs.append({
            'name': cname,
            'xmin': int(xmin),
            'ymin': int(ymin),
            'xmax': int(xmax),
            'ymax': int(ymax)
        })

    return json.dumps({"objects": detect_objs})


if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('/usr/local/ev_sdk/data/dog.jpg')
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)
