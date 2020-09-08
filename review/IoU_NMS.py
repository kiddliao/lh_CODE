import torch
import numpy as np
import math

def IoU(bbox1, bbox2, type='IoU'):
    '''
    bbox1.shape=(n,4),bbox2.shape=(m,4)
    '''
    bbox1.unsqueeze_(dim=-1)
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]
    inter_area_w = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    inter_area_h = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b1_y1)
    #保证不相交的俩框相交面积是0 而不会出现长宽为负
    inter_area = torch.clamp(inter_area_w, min=0.0) * torch.clamp(inter_area_h, min=0.0) 
    
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area

    if type == 'IoU':
        return iou
    else:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if type == 'GIoU':
            c_area = cw * ch + 1e-16
            return iou - (c_area - union_area) / c_area
        if type in ['DIoU', 'CIoU']:
            c2 = cw ** 2 + ch ** 2 + 1e-16  #勾股定理 对角线长度
            rho2 = ((b2_x2 + b2_x1) / 2 - (b1_x2 + b1_x1) / 2)** 2 + \
                ((b2_y2 + b2_y1) / 2 - (b1_y2 + b1_y1) / 2)** 2 #中心点的欧式距离
            if type == 'DIoU':
                return iou - rho2 / c2
            elif type == 'CIoU':
                w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
                w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
                v = (4 / math.pi ** 2) * (torch.pow(torch.atan(w1 / h1) - torch.atan(w2 / h2)))
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)


def NMS(predictions, conf_thresh=0.05, iou_thresh=0.5, style='OR', type='iou'):
    #传入的是1张图片的1个分类的预测结果
    if torch.sum(predictions[:, 4] >= conf_thresh) == torch.tensor(0):
        return []
    #删除置信度小于阈值的预测框
    conf_mask = predictions[:, 4] >= conf_thresh
    predictions = predictions[conf_mask]
    
    indices = torch.argsort(-1 * predictions[:, 4])
    predictions = predictions[indices]
    
    result = []
    dc = predictions.clone()
    
    if style == 'OR':
        while dc.size()[0]:
            result.append(dc[:1].squeeze().tolist())
            if dc.size()[0] == 1:
                break
            iou = IoU(dc[1:], dc[:1], type='IoU')
            save_indices = (iou < iou_thresh)
            dc = dc[1:][save_indices]
    
    elif style == 'AND':
        while dc.size()[0] > 1:
            iou = IoU(dc[1:], dc[:1], type='IoU')
            save_indices = (iou < iou_thresh)
            if iou.max() >= iou_thresh:
                result.append(dc[:1].squeeze().tolist())
            dc = dc[1:][save_indices]

    elif style == 'SOFT':
        sigma = 0.5
        while dc.size()[0]:
            if dc.size()[0] == 1:
                result.append(dc[:1].squeeze().tolist())
                break
            result.append(dc[:1].squeeze().tolist())
            iou = IoU(dc[1:], dc[:1], type='IoU')
            dc = dc[1:]
            dc[:, 4] *= torch.exp(-iou ** 2 / sigma)
            save_indices = (dc[:,4]>=conf_thresh)
            dc = dc[1:][save_indices]

    elif style == 'Merge':
        while dc.size()[0]:
            if dc.size()[0] == 1:
                result.append(dc[:1].squeeze().tolist())
                break
            iou = IoU(dc[1:], dc[:1], type='IoU')
            repeat_indices = (iou >= iou_thresh)
            weights = dc[repeat_indices, 4:5]
            dc[0,:4] = (weights * dc[repeat_indices,:4]).sum(0) / weights.sum()
            result.append(dc[:1].squeeze().tolist())
            dc = dc[1:][~repeat_indices]
    
    return result
            
    
    
    
