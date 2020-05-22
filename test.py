# import json
# import os
# # os.chdir(os.path.join('..', 'REMOTE', 'datasets', 'coco_xray'))

# # with open(os.path.join('annotations', 'newinstances_test2017.json'), 'r') as f:
# with open('new_val.json') as f:
#     res = json.load(f)
# pass
import torch
def lh_IOU(box1, box2, x1y1x2y2=False, iou_type='IoU'):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # else:  # x, y, w, h = box1
    #     b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    #     b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    #     b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    #     b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:  # x1, y1, w, h = box1
            b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
            b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
            b2_x1, b2_x2 = box2[0], box2[0] + box2[2] 
            b2_y1, b2_y2 = box2[1], box2[1] + box2[3] 

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    iou = inter_area / union_area  # iou
    if iou_type in ['GIoU', 'CIoU', 'DIoU']:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if iou_type=='GIoU':  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
        if iou_type in ['CIoU','DIoU']:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if iou_type=='DIoU':
                return iou - rho2 / c2  # DIoU
            elif iou_type=='CIoU':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou
print(lh_IOU(torch.tensor([1,2,0.9,0.9]),torch.tensor([[1,2,0.9,0.9]])))