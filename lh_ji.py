from __future__ import print_function
import logging as log
import json
import os
import cv2
import numpy as np

import argparse
import os
import warnings

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

log.basicConfig(level=log.DEBUG)
os.chdir(os.path.join("/usr","local",'ev_sdk',"mmdetection"))

class_name = {
    1: "vendors"
}


def init():
    config = '/usr/local/ev_sdk/faster_resnet50.py'
    checkpoint = '/usr/local/ev_sdk/model/epoch_24.pth'
    model = init_detector(config, checkpoint, device='cuda:0')
    log.info('Loading model...')
    return model
    


def process_image(model, img, args=None):
    result = inference_detector(model, img)
    detect_objs = []
    #单分类 只有一个vendors
    for cid,cname in class_name.items():
        c_index = cid - 1
        for box in result[c_index]:
            xmin, ymin, xmax, ymax, conf = box
            if min(xmin, ymin, xmax, ymax) > 1.0 and conf > 0.05:
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
