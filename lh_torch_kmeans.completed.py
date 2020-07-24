import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, cluster_number, path, file_name, colors):
        self.cluster_number = cluster_number
        self.path = path
        self.file_name = file_name
        self.colors = colors
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def IOU(self, boxes, clusters):
        n = boxes.size()[0]
        k = clusters.size()[0]
        boxes = boxes.unsqueeze(-1)

        boxes_area = boxes[:, 0] * boxes[:, 1]
        clusters_area = clusters[:, 0] * clusters[:, 1]
        #将每个框的中心点放在(0,0)
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = -boxes[:, 0] / 2, -boxes[:, 1] / 2, boxes[:, 0] / 2, boxes[:, 1] / 2
        clusters_x1, clusters_y1, clusters_x2, clusters_y2 = -clusters[:,
                                                                       0] / 2, -clusters[:,
                                                                                         1] / 2, clusters[:,
                                                                                                          0] / 2, clusters[:,
                                                                                                                           1] / 2
        inter_area=(torch.min(boxes_x2,clusters_x2)-torch.max(boxes_x1,clusters_x1)).clamp(min=0) *\
                (torch.min(boxes_y2,clusters_y2)-torch.max(boxes_y1,clusters_y1)).clamp(min=0)
        union_area = boxes_area + clusters_area - inter_area + 1e-16
        iou = inter_area / union_area
        return iou

    def get_width_height(self, file_path):
        with open(file_path, 'r') as f:
            labels = f.readlines()
        labels = list(filter(lambda x: x[-1] != '', list(map(lambda x: x.strip().split(' '), labels))))
        boxes = []
        for label in labels:
            w, h = int(label[1]), int(label[2])
            for i in label[3:]:
                i = i.split(',')
                width = float(i[2]) * w
                height = float(i[3]) * h
                boxes.append([width, height])
        boxes = torch.Tensor(boxes).to(self.device)
        return boxes

    def kmeans(self, boxes):
        box_number = boxes.shape[0]
        last_nearest = torch.zeros(box_number).to(self.device)
        # np.random.seed(42)
        #初始化k个中心点
        clusters = boxes[np.random.choice(box_number, self.cluster_number, replace=False)]  #replace代表不许重复

        # plt.scatter(boxes[...,0],boxes[...,1],color='yellow',alpha=1,s=5)
        # plt.scatter(clusters[:,0],clusters[:,1],color='black',alpha=1,s=100)
        # plt.show()

        while True:
            iou = self.IOU(boxes, clusters)
            iou_max, iou_argmax = torch.max(iou, dim=1)
            if (last_nearest == iou_argmax).all():
                break
            last_nearest = iou_argmax.clone()
            for i in range(self.cluster_number):
                clusters[i, :] = torch.median(boxes[iou_argmax == i, :], dim=0).values
                # plt.scatter(clusters[i:i+1,0],clusters[i:i+1,1],color=colors[i],alpha=1,s=100)
                # plt.scatter(boxes[iou_argmax==i,0],boxes[iou_argmax==i,1],color=colors[i],alpha=1,s=5)
            # plt.show()
        indices = torch.arange(9).to(self.device).unsqueeze(0).repeat(box_number, 1)
        map_indices = (indices == last_nearest.unsqueeze(-1))
        iou = iou[map_indices]
        acc = iou.mean().item()
        clusters = clusters.tolist()
        clusters.sort(key=lambda x: (x[0], x[1]))
        for i in clusters:
            print(i)
        print('accuracy is {}%'.format(round(acc, 4) * 100))
        return clusters

    def save_results(self, anchor):
        with open(os.path.join(self.path, 'anchor.txt'), 'w') as f:
            anchor = list(map(lambda x: ','.join(list(map(str, x))), anchor))
            tmp = ', '.join(anchor)
            f.write(tmp + '\n')

    def core(self):
        boxes = self.get_width_height(os.path.join(self.path, self.file_name))
        # plt.scatter(boxes[...,0],boxes[...,1],color='yellow',alpha=1,s=10)
        # plt.show()
        anchor = self.kmeans(boxes)
        self.save_results(anchor)


if __name__ == '__main__':
    cluster_number = 9
    colors = ['red', 'pink', 'blue', 'cyan', 'green', 'gray', 'orange', 'gold', 'purple']
    path = os.path.join('/home', 'lh', 'myhome', 'datasets', 'coco_shape')
    file_name = 'train_all_labels.txt'
    with open(os.path.join(path, '2007_train.txt'), 'r') as f:
        train = f.readlines()
    with open(os.path.join(path, '2007_val.txt'), 'r') as f:
        val = f.readlines()
    train = train + val
    train = list(map(lambda x: x.strip(), train))

    # with open(os.path.join(path, file_name), 'w') as f:
    #     for i in train:
    #         image_path = i
    #         image_name = os.path.basename(image_path).split('.')[0]
    #         pic = cv2.imread(os.path.join(path, 'JPEGImages', image_name + '.jpg'))
    #         h, w, _ = pic.shape
    #         with open(os.path.join(path, 'labels', image_name + '.txt'), 'r') as f2:
    #             labels = f2.readlines()
    #             cur_labels = []
    #             for j in range(len(labels)):
    #                 label = labels[j].strip().split()
    #                 tmp = label[1:] + [label[0]]
    #                 tmp = ','.join(tmp)
    #                 cur_labels.append(tmp)
    #             write_in = [image_path, str(w), str(h)] + cur_labels
    #             write_in = ' '.join(write_in)
    #             f.write(write_in + '\n')

    kmeans = KMeans(cluster_number, path, file_name, colors)
    kmeans.core()