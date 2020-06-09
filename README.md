## **cv目标检测脚本(自己写的)**

1.数据集转换、划分、统计信息脚本(比如lh_voc_xxx_2_coco_xxx.py)

常见的数据集有下列几种
coco格式(标注为json)|voc格式(标注为xml)|voc格式(标注为txt)|tfrecord格式
-------------------|:---------------:|----------------:|-:
----
常见的预测文件有下列几种
coco格式(json)  |darknet格式(classes.txt)
-:|:-

2.数据集可视化脚本(比如lh_coco_label_imshow.py)
支持coco格式和voc格式的可视化

3.多模型结果模型融合脚本和四种常见nms算法实现(比如lh_pred_nms_to_best.py)

4.pytorch读取数据集的pipeline脚本(比如lh_torch_datasets_pipeline.py)

5.力扣算法(比如lh_algorithm_xxx.py)

6.目标检测评估脚本(比如lh_coco_eval.py)

7.多种学习率衰减的pytorch实现脚本(比如lh_implements_lr_decay_schedule.py)

8.损失函数注释(比如lh_focal_loss_and_smooth_l1_loss.py)