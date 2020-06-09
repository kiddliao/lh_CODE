import sys
import cv2
from random import randint

#使用步骤：
#pip install opencv-contrib-python opencv-python
#1、在所要跟踪的物体上画矩形框，再按除‘q’和‘Esc’外的任意键，即可选中跟踪目标；
#2、重复步骤1即可选择多个跟踪目标；
#3、按q键开始跟踪；
#4、按Esc键退出运行。


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
    # 通过跟踪器的名字创建跟踪器
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available tracker name')
        
        for t in trackerTypes:
            print(t)
    return tracker
 
print('Default tracking algorithm is CSRT \n'
     'Available tracking algorithms are:\n')
for t in trackerTypes:
    print(t)
    
trackerType = 'CSRT'
 
# 设置加载的视频文件
videoPath = 'run.mp4'
 
# 创建video capture 来读取视频文件
cap = cv2.VideoCapture(videoPath)
 
# 读取第一帧
ret, frame = cap.read()
 
# 如果无法读取视频文件就退出
if not ret:
    print('Failed to read video')
    sys.exit(1)
    
# 选择框
bboxes = []
colors = []
 
# OpenCV的selectROI函数不适用于在Python中选择多个对象
# 所以循环调用此函数，直到完成选择所有对象
while True:
    # 在对象上绘制边界框selectROI的默认行为是从fromCenter设置为false时从中心开始绘制框，可以从左上角开始绘制框
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) 
    if (k == 113):  # q is pressed
        break
        
print('Selected bounding boxes {}'.format(bboxes))
# 初始化MultiTracker
# 有两种方法可以初始化multitracker
# 1. tracker = cv2.MultiTracker（“CSRT”）
# 所有跟踪器都添加到这个多路程序中
# 将使用CSRT算法作为默认值
# 2. tracker = cv2.MultiTracker（）
# 未指定默认算法
# 使用跟踪算法初始化MultiTracker
# 指定跟踪器类型
 
# 创建多跟踪器对象
multiTracker = cv2.MultiTracker_create()
 
# 初始化多跟踪器
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    
# 处理视频并跟踪对象
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 获取后续帧中对象的更新位置
    ret, boxes = multiTracker.update(frame)
    
    # 绘制跟踪的对象
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
 
    # show frame
    cv2.imshow('MultiTracker', frame)
    
 
    # quit on ESC button
    if cv2.waitKey(1) == 27:  # Esc pressed
        break