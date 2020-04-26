import torch
import torch.nn as nn
import cv2
import numpy as np
#实现focal loss
def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    #a.shape=(n,4),a[:,2].shape=(n),torch.unsqueeze(a[:,2],dim=1).shape=(n,1)
    area_a = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
    xa1, ya1 = torch.unsqueeze(a[:, 1], dim=1), torch.unsqueeze(a[:, 0], dim=1)
    xa2, ya2 = torch.unsqueeze(a[:, 3], dim=1), torch.unsqueeze(a[:, 2], dim=1)
    xb1, yb1, xb2, yb2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    iw = torch.min(xa2, xb2) - torch.max(xa1, xb1)
    ih = torch.min(ya2, yb2) - torch.max(ya1, yb1)
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih #(n,1)
    IOU = intersection / (area_a + area_b - intersection)
    return IoU


class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model

    def forward(self, imgs, annotations):
        _, regression, classification, anchors = self.model(imgs)
        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss

class FocalLoss(nn.Module):
    def __init__(self):
        #首先找到FocalLoss的父类nn.Module,然后把类FocalLoss的对象转换为nn.Module的对象
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        # anchor是由不同尺度的每个网格的中心点和每个网格对应的几个anchor框而计算出来的目标框
        # 假如第i个网格的anchor框和gt的第一个框IOU最大 那么代表该网格能不错的预测这个gt框
        # 那么就用这个网格的这个anchor框去预测这个gt框
        # 之后计算分类损失也对该gt框的类别计算损失


        # anchor是所有尺度的所有网格预测的框
        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(torch.tensor(0).to(dtype))

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            # 计算每个网格预测的目标框和每个gt的IOU 
            # 如果p3尺度的(i,j)网格预测的目标框(x1,y1,x2,y2)在所有的5个gt框种与第1个gt的IOU最大
            # 那么后面就当该网格的目标框预测的是第一个gt框的类
            # anchor (n,4) bbox_annotation(i,5) IOU(n,i) 代表每个网格预测的框和每个gt框的IOU
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
            # IOU_argmax就是用那个gt框代表这个网格预测的框
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            # 得到一个(n,1)的全为-1的矩阵 -1表示的是初始状态位 n是所有尺度的图片的grid总数
            targets = torch.ones_like(classification) * -1 #(n,3) 3是分类的总数 coco是80
            if torch.cuda.is_available():
                targets = targets.cuda()
            # torch.lt 小于第二个参数为True torch.ge大于第二个参数为True
            targets[torch.lt(IoU_max, 0.4), :] = 0 #IOU低于0.4直接置0 即该框为背景不参与分类的损失计算

            positive_indices = torch.ge(IoU_max, 0.5) #IOU高于0.5为True 在下面置1 即该框为前景参与损失计算
            # 此时target矩阵种有两种状态位 -1和0

            # 计算此时所有预测为正例的个数
            num_positive_anchors = positive_indices.sum()
            # bbox_annotation是当前图片的gt框 
            # assigned_annotations (n,5) 即每个网格预测的框对应的gt框
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            # iou大于阈值的框的target先置全部0 其他的框还全是-1和0
            targets[positive_indices,:] = 0  #(n,3)
            # assigned_annotations[positive_indices, 4]是成功预测了前景框的的网格预测的类别号 这里类似生成了one-hot码
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha  #(n,3) 3是类别数
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()
            # torch.eq(targets, 1.) (n,3) 
            # 预测为正的样本的正负样本平衡因子为(1-α)即0.75 预测为负的样本的正负样本平衡因子为α 0.25 减小了正样本和负样本的不均衡带来的影响
            # alpha_factor是平衡因子矩阵 focal_weight是预测值y'矩阵
            # 然后次方乘gamma算出来了focal_weight即(1-α)y'^γ和α(1-y)'^γ
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            # bce即二分类交叉熵损失 -log(y')+log(1-y')
            # targets (n,3) 已经是只有大于iou的阈值的框对应的类别号是1的矩阵 其他值都是0或-1 下面通过矩阵相乘巧妙实现了bce
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            # cls_loss就是最后算出来的focal_loss了
            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            # torch.ne 不等于index的为True 通过torch.where 将-1项置0
            # 即IOU处在0.4和0.5之间的框既不作为正样本也不作为负样本计入损失
            # 而IOU小于0.4的框之前已经置0作为前景计算了对每个框预测的所有的3个类的概率计算了log(1-y')
            # IOU大于0.5的框为独热码,预测为正样本的计算log(y'),预测为负样本的计算log(1-y')
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros) 
            # classification_losses为该batch所有图片的分类损失
            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:  #如果有正样本就计算回归损失 也就是yolov3里的置信度超过某个值就认为有物体 才对其进行预测和计算回归损失
                # assigned_annotations现在是iou大于阈值的目标框对应的gt框
                assigned_annotations = assigned_annotations[positive_indices, :]
                # 转换为中心点坐标和宽高的格式
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                # yolov3的损失变动? 回归网络的任务就是得到回归的偏移量而不是直接得到数据
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()
                # 实际的偏移量是gt框相对于落入的网格的左上角的坐标偏移和宽高偏移
                # 预测的偏移就是经过神经网络计算的(x,4)维特征矩阵
                # 预测的偏移量和实际的偏移量之间的差距
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                # 分段损失函数 连续且可导 l1正则化
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)

        