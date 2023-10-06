 
# yolov8快速使用说明
 
# python -m venv venv
# source venv/Scripts/activate
# pip install ultralytics
# python main.py
 
import os
import cv2 # 导入cv2库
from ultralytics import YOLO # 导入ultralytics库中的YOLO模块
import numpy as np # 导入numpy库
 
 
cap = cv2.VideoCapture("./img/physics.jpg") # 创建一个视频捕获对象
# cap = cv2.VideoCapture("E:/BaiduNetdiskDownload/我想吃掉你的胰脏  念念手纪/念念手纪/我想吃了你的胰脏.2017.BD1080P.日语中字.mp4") # cv2.VideoCapture是一次性读取视频的，如果视频过大，会导致内存不足，程序崩溃
if not cap.isOpened(): # 判断是否正常打开
    print("视频打开失败")
    exit()
 
model = YOLO("./models/yolo/yolov8m.pt") # 加载预训练好的YOLOv8模型

while True: # 循环地处理每一帧图像
 
    ret, frame = cap.read() # 从视频捕获对象中读取一帧图像
    if not ret: # 如果没有读取到图像，说明视频已经结束，跳出循环
        break
    # 配置使用mac book 的gpu加速(mps)
    # results = model(frame, device="mps") # 将图像输入到YOLOv8模型中，得到检测结果
    # cuda 12.2
    results = model(frame, device=0) # 将图像输入到YOLOv8模型中，得到检测结果
    result = results[0] # 取出第一个检测结果（如果有多个结果，表示有多个图像输入）
 
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confidences = np.array(result.boxes.conf.cpu(), dtype="float")

    for cls, bbox, confidence in zip(classes, bboxes, confidences):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        label = f"{model.names[cls]} {confidence:.2f}"
        # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - label_height - baseline), (x + label_width, y), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
 
    cv2.namedWindow("Img", cv2.WINDOW_NORMAL) # 创建一个可以调整大小的窗口
    cv2.imshow("Img", frame) # 显示处理后的图像
    key = cv2.waitKey(1) # 等待用户按键
    if key == 27: # 如果用户按下ESC键，退出程序
        break
 
cap.release() # 释放视频捕获对象
cv2.destroyAllWindows() # 销毁所有窗口
 
 
'''
A class for storing and manipulating detection boxes.

    Args:
        boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes,
            with shape (num_boxes, 6) or (num_boxes, 7). The last two columns contain confidence and class values.
            If present, the third last column contains track IDs.
        orig_shape (tuple): Original image size, in the format (height, width).

    Attributes:
        xyxy (torch.Tensor | numpy.ndarray): The boxes in xyxy format.
        conf (torch.Tensor | numpy.ndarray): The confidence values of the boxes.
        cls (torch.Tensor | numpy.ndarray): The class values of the boxes.
        id (torch.Tensor | numpy.ndarray): The track IDs of the boxes (if available).
        xywh (torch.Tensor | numpy.ndarray): The boxes in xywh format.
        xyxyn (torch.Tensor | numpy.ndarray): The boxes in xyxy format normalized by original image size.
        xywhn (torch.Tensor | numpy.ndarray): The boxes in xywh format normalized by original image size.
        data (torch.Tensor): The raw bboxes tensor (alias for `boxes`).

    Methods:
        cpu(): Move the object to CPU memory.
        numpy(): Convert the object to a numpy array.
        cuda(): Move the object to CUDA memory.
        to(*args, **kwargs): Move the object to the specified device.
'''