# This Python file uses the following encoding: utf-8
import sys
import cv2
import time
import numpy as np
from PySide2.QtWidgets import QApplication, QWidget, QGraphicsScene, QFileDialog,QMessageBox,QInputDialog, QLabel
from PySide2.QtGui import QPixmap, QImage, QMovie
from PySide2.QtCore import QTimer, Qt

from ui import Ui_Widget
from image_processing import *
# from capture import *

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.startAnimation()
        self.init_button()
        self.init_val()

    def startAnimation(self):
        # 启动动画
        self.movie = QMovie("./resources/output.gif")
        if self.movie is None:
            print("Error: Failed to load animation.")
            return
        self.label_anime = QLabel(self)
        self.label_anime.setAlignment(Qt.AlignCenter)
        self.label_anime.setMovie(self.movie)
        # self.label_anime.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        # self.label_anime.setAttribute(Qt.WA_TranslucentBackground)
        # self.label_anime.setGeometry(self.geometry())
        # self.label_anime.setScaledContents(True)
        self.movie.start()
        self.label_anime.showFullScreen()
        # 计算动画时长
        duration = self.movie.frameCount() * self.movie.nextFrameDelay()

        # 在动画播放完毕后关闭动画
        QTimer.singleShot(duration, self.stopAnimation)
    
    def stopAnimation(self):
        # 停止动画
        if self.movie:
            self.movie.stop()
            self.label_anime.hide()
        del self.movie
        del self.label_anime
    
    def resizeEvent(self, event):
        if self.label_anime is not None:
            # 窗口大小改变时，调整QLabel大小
            self.label_anime.setGeometry(self.label_anime.geometry().x(), self.label_anime.geometry().y(), self.width(), self.height())
            self.label_anime.setScaledContents(True)
    
    def init_button(self):
        # Connect the buttons
        # block 1
        self.ui.pushButton_37.clicked.connect(self.button_37_clicked) # open camera
        self.ui.pushButton_24.clicked.connect(self.button_24_clicked) # close camera
        self.ui.pushButton_34.clicked.connect(self.button_34_clicked) # capture image
        self.ui.pushButton.clicked.connect(self.button_clicked) # Load the image
        self.ui.pushButton_2.clicked.connect(self.button_2_clicked) # save the image
        self.ui.ClearButton.clicked.connect(self.clear_windows) # reset the image

        # block 2
        self.ui.pushButton_7.clicked.connect(self.button_7_clicked) # zoom in
        self.ui.pushButton_15.clicked.connect(self.button_15_clicked) # zoom out
        self.ui.pushButton_38.clicked.connect(self.button_38_clicked) # resize
        self.ui.pushButton_38.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_38.objectName()))
        self.ui.pushButton_36.clicked.connect(self.button_36_clicked) # Rotate
        self.ui.pushButton_36.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_36.objectName()))
        self.ui.pushButton_35.clicked.connect(self.button_35_clicked) # flip
        self.ui.pushButton_35.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_35.objectName()))
        self.ui.pushButton_4.clicked.connect(self.button_4_clicked) # Histogram
        self.ui.pushButton_4.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_4.objectName()))

        # Process
        self.ui.pushButton_3.clicked.connect(self.button_3_clicked) # grayscale
        self.ui.pushButton_3.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_3.objectName()))
        self.ui.pushButton_5.clicked.connect(self.button_5_clicked) # Histogram Normalization
        self.ui.pushButton_5.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_5.objectName()))
        self.ui.pushButton_6.clicked.connect(self.button_6_clicked) # Morphological Diff
        self.ui.pushButton_6.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_6.objectName()))
        self.ui.pushButton_19.clicked.connect(self.button_19_clicked) # Gaussian Noise
        self.ui.pushButton_19.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_19.objectName()))
        self.ui.pushButton_18.clicked.connect(self.button_18_clicked) # Salt and Pepper Noise
        self.ui.pushButton_18.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_18.objectName()))

        # Filter
        self.ui.pushButton_13.clicked.connect(self.button_13_clicked) # Mean Filter
        self.ui.pushButton_13.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_13.objectName()))
        self.ui.pushButton_14.clicked.connect(self.button_14_clicked) # Median Filter
        self.ui.pushButton_14.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_14.objectName()))
        self.ui.pushButton_16.clicked.connect(self.button_16_clicked) # Gaussian Filter
        self.ui.pushButton_16.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_16.objectName()))
        self.ui.pushButton_17.clicked.connect(self.button_17_clicked) # Bilateral Filter
        self.ui.pushButton_17.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_17.objectName()))

        # Edge Detection
        self.ui.pushButton_8.clicked.connect(self.button_8_clicked) # Sobel
        self.ui.pushButton_8.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_8.objectName()))
        self.ui.pushButton_9.clicked.connect(self.button_9_clicked) # Roberts
        self.ui.pushButton_9.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_9.objectName()))
        self.ui.pushButton_10.clicked.connect(self.button_10_clicked) # Prewitt
        self.ui.pushButton_10.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_10.objectName()))
        self.ui.pushButton_11.clicked.connect(self.button_11_clicked) # Laplacian
        self.ui.pushButton_11.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_11.objectName()))
        self.ui.pushButton_12.clicked.connect(self.button_12_clicked) # Canny
        self.ui.pushButton_12.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_12.objectName()))
        # 连接spinBox_2与spinBox_3的valueChanged信号与check_spinbox_values槽函数
        self.ui.spinBox_2.valueChanged.connect(self.check_spinbox_values)
        self.ui.spinBox_3.valueChanged.connect(self.check_spinbox_values)

        # Image Segmentation
        self.ui.pushButton_25.clicked.connect(self.button_25_clicked) # Adaptive Threshold
        self.ui.pushButton_25.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_25.objectName()))
        self.ui.pushButton_26.clicked.connect(self.button_26_clicked) # Otsu Threshold
        self.ui.pushButton_26.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_26.objectName()))
        self.ui.pushButton_27.clicked.connect(self.button_27_clicked) # Watershed
        self.ui.pushButton_27.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_27.objectName()))
        self.ui.pushButton_28.clicked.connect(self.button_28_clicked) # Background Subtractor(MOG2)
        self.ui.pushButton_28.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_28.objectName()))
        self.ui.pushButton_29.clicked.connect(self.button_29_clicked) # Background Subtractor(KNN)
        self.ui.pushButton_29.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_29.objectName()))
        
        # Objective Detection
        self.ui.pushButton_30.clicked.connect(self.button_30_clicked) # YOLOv5
        self.ui.pushButton_30.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_30.objectName()))
        self.ui.pushButton_31.clicked.connect(self.button_31_clicked) # YOLOv8
        self.ui.pushButton_31.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_31.objectName()))
        self.ui.pushButton_32.clicked.connect(self.button_32_clicked) # Face Detection(cascade)
        self.ui.pushButton_32.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_32.objectName()))
        self.ui.pushButton_33.clicked.connect(self.button_33_clicked) # SIFT
        self.ui.pushButton_33.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_33.objectName()))
        self.ui.pushButton_20.clicked.connect(self.button_20_clicked) # SURF
        self.ui.pushButton_20.clicked.connect(lambda: self.update_button_state(self.ui.pushButton_20.objectName()))

    def init_val(self):

        #graph views
        self.scene, self.scene_2 = QGraphicsScene(), QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.show()
        self.ui.graphicsView_2.setScene(self.scene_2)
        self.ui.graphicsView_2.show()

        # image buffer
        self.image = None
        self.img_path = None
        self.img_copy = None

        # kernel size
        self.kernel_size = 3

        # Open the camera
        self.camera_opened = False  # Flag to avoid camera openeing more than once
        self.cap = None # VideoCapture object
        self.VideoPath = 0 # Video path
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame) # Connect timeout signal to update_frame slot
        self.timer.timeout.connect(self.keep_processing)
        self.start_time = 0
        
        # Background Subtractor
        self.bg_subtractor_MOG2 = cv2.createBackgroundSubtractorMOG2()
        self.bg_subtractor_KNN = cv2.createBackgroundSubtractorKNN()

        # Face Detection
        self.face_cascade_path = "./models/haarcascades/haarcascade_frontalface_alt2.xml"
        self.face_cascade = cv2.CascadeClassifier()

        # SIFT
        self.sift = cv2.SIFT_create()

        # FAST
        self.fast = cv2.FastFeatureDetector_create()

        # YOLO
        self.yolov5_model = YOLO('models/yolo/yolov5su.pt') # 加载预训练的 YOLOv5s 模型
        self.yolov8_model = YOLO('models/yolo/yolov8m.pt') # 加载预训练的 YOLOv8m 模型

        # 持续处理图像
        self.keep_processing_flag = False # Flag to keep processing image (used for 'open camera' button, 'close camera' button), press 'capture image' button to stop processing
        self.last_button = None # 上一次按下的按钮
        self.button_func_dict = dict([ # 按钮与对应函数的字典, 被注释说明不支持camrea_opened=True时的实时处理
        (self.ui.pushButton_38.objectName(), self.button_38_clicked), # resize
        (self.ui.pushButton_36.objectName(), self.button_36_clicked), # Rotate
        (self.ui.pushButton_35.objectName(), self.button_35_clicked), # flip
        # (self.ui.pushButton_4.objectName(), self.button_4_clicked), # Histogram
        (self.ui.pushButton_3.objectName(), self.button_3_clicked), # grayscale
        (self.ui.pushButton_5.objectName(), self.button_5_clicked), # Histogram Normalization
        (self.ui.pushButton_6.objectName(), self.button_6_clicked), # Morphological Diff
        (self.ui.pushButton_19.objectName(), self.button_19_clicked), # Gaussian Noise
        (self.ui.pushButton_18.objectName(), self.button_18_clicked), # Salt and Pepper Noise
        (self.ui.pushButton_13.objectName(), self.button_13_clicked), # Mean Filter
        (self.ui.pushButton_14.objectName(), self.button_14_clicked), # Median Filter
        (self.ui.pushButton_16.objectName(), self.button_16_clicked), # Gaussian Filter
        (self.ui.pushButton_17.objectName(), self.button_17_clicked), # Bilateral Filter
        (self.ui.pushButton_8.objectName(), self.button_8_clicked), # Sobel
        (self.ui.pushButton_9.objectName(), self.button_9_clicked), # Roberts
        (self.ui.pushButton_10.objectName(), self.button_10_clicked), # Prewitt
        (self.ui.pushButton_11.objectName(), self.button_11_clicked), # Laplacian
        (self.ui.pushButton_12.objectName(), self.button_12_clicked), # Canny
        (self.ui.pushButton_25.objectName(), self.button_25_clicked), # Adaptive Threshold
        (self.ui.pushButton_26.objectName(), self.button_26_clicked), # Otsu Threshold
        (self.ui.pushButton_27.objectName(), self.button_27_clicked), # Watershed
        (self.ui.pushButton_28.objectName(), self.button_28_clicked), # Background Subtractor(MOG2)
        (self.ui.pushButton_29.objectName(), self.button_29_clicked), # Background Subtractor(KNN)
        (self.ui.pushButton_30.objectName(), self.button_30_clicked), # YOLOv5
        (self.ui.pushButton_31.objectName(), self.button_31_clicked), # YOLOv8
        (self.ui.pushButton_32.objectName(), self.button_32_clicked), # Face Detection(AdaBoost)
        (self.ui.pushButton_33.objectName(), self.button_33_clicked), # SIFT
        (self.ui.pushButton_20.objectName(), self.button_20_clicked), # FAST
        ])
     
    # open camera
    def button_37_clicked(self):
        if self.camera_opened:
            return
        print("Open Camera")
        self.clear_all_windows()
        # self.cap = RTSCapture(0) # Open the camera
        # 弹窗选择打开方式，包含打开摄像头、打开视频文件、打开图片文件
        items = ("Camera", "IP Camera", "Video")
        item, ok = QInputDialog.getItem(self, "Select", "Open", items, 0, False)
        if ok and item:
            if item == "Camera":
                self.VideoPath = 0
            elif item == "IP Camera":
                self.VideoPath, ok = QInputDialog.getText(self, "Input", "IP Address", text="http://")
            elif item == "Video":
                self.VideoPath, ok = QFileDialog.getOpenFileName(self, 'Open file', filter="Video files (*.mp4 *.avi *.mkv) ")
        self.cap = cv2.VideoCapture(self.VideoPath) # Open the camera, default 0
        if not self.cap.isOpened():
            QMessageBox.information(self,"Warning","Open Camera Failed.")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", self.fps)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # MJPG
        self.camera_opened = True
        self.keep_processing_flag = True
        # self.timer = QTimer() # Create Timer object
        # self.timer.timeout.connect(self.update_frame) # Connect timeout signal to update_frame slot
        # self.timer.timeout.connect(self.keep_processing)
        # self.timer.start(1) # Emit the timeout signal at x ms interval
        self.timer.start(1000/self.fps) # Use the fps of the camera
    
    #close camera
    def button_24_clicked(self):
        if not self.camera_opened:
            return
        print("Close Camera")
        self.keep_processing_flag = False
        self.camera_opened = False
        self.timer.stop() # Stop the timer
        self.cap.release() # Release the camera
        self.scene.clear() # Clear graphics scene
    
    # capture image
    def button_34_clicked(self):
        # Capture image from the camera
        if not self.camera_opened or self.img_copy is None:
            return
        print("Capture Image")
        # ret, image = self.cap.read()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.keep_processing_flag = False
        self.last_button = None
        self.img_copy = self.image.copy()
        self.display_image(self.img_copy, self.scene_2)

    # load image from file
    def button_clicked(self):
        # Load the image and show it in the graphics view
        self.img_path = self.__load_image_path()
        if self.img_path is None:
            return
        # 判断相机是否打开
        if self.camera_opened:
            self.button_24_clicked() # 关闭相机
        self.clear_all_windows()
        self.image = cv2.imread(self.img_path)
        self.img_copy = self.image.copy()
        self.display_image(self.img_copy, self.scene)

    # save image to file
    def button_2_clicked(self):
        # Save the image
        if self.img_copy is None:
            QMessageBox.information(self,"Warning","No image to save.")
            return
        self.last_button = None
        img_save_path, _ = QFileDialog.getSaveFileName(self, 'Save file', filter="Image files (*.jpg *.jpeg *.bmp *.png) ") #pylint: disable=line-too-long
        if img_save_path == '':
            QMessageBox.information(self,"Warning","No file selected.")
            return
        else:
            cv2.imwrite(img_save_path, self.img_copy)
            QMessageBox.information(self,"Information","Image saved successfully.")
            return

    # zoom in
    def button_7_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to zoom in.")
            return
        # 缩放倍数
        zoom_rate = 1.25
        # 视窗缩放
        self.ui.graphicsView.scale(zoom_rate, zoom_rate)
        self.ui.graphicsView_2.scale(zoom_rate, zoom_rate)
    
    # zoom out
    def button_15_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to zoom out.")
            return
        # 缩放倍数
        zoom_rate = 0.8
        # 视窗缩放
        self.ui.graphicsView.scale(zoom_rate, zoom_rate)
        self.ui.graphicsView_2.scale(zoom_rate, zoom_rate)

    # resize image
    def button_38_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to zoom in.")
            return
        # 从LineEdit中获取缩放倍数
        zoom_rate = self.ui.lineEdit.text()
        if zoom_rate == '':
            QMessageBox.information(self,"Warning","No zoom rate input.")
            return
        zoom_rate = float(zoom_rate)
        # 判断缩放倍数是否合法或是否过大
        if zoom_rate <= 0:
            QMessageBox.information(self,"Warning","Zoom rate should be positive.")
            return
        elif zoom_rate > 10:
            QMessageBox.information(self,"Warning","Zoom rate should be less than 10.")
            return
        # 缩放
        self.img_copy = cv2.resize(self.img_copy, (0,0), fx=zoom_rate, fy=zoom_rate, interpolation=cv2.INTER_CUBIC)
        self.display_image(self.img_copy, self.scene_2)
        # 视窗缩放
        # self.ui.graphicsView.scale(zoom_rate, zoom_rate)

    # rotate
    def button_36_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to rotate.")
            return
        # 从LineEdit_2中获取旋转角度
        rotate_angle = self.ui.lineEdit_2.text()
        if rotate_angle == '':
            QMessageBox.information(self,"Warning","No rotate angle input.")
            return
        rotate_angle = float(rotate_angle)
        rotate_angle = rotate_angle % 360
        # 判断旋转角度是否合法
        # if rotate_angle < 0 or rotate_angle > 360:
        #     QMessageBox.information(self,"Warning","Rotate angle should be in [0, 360].")
        #     return
        # 旋转
        if rotate_angle == 0:
            # 旋转角度为0时，不做任何处理
            self.display_image(self.img_copy, self.scene_2)
        elif rotate_angle % 90 == 0:
            # 分别处理旋转角度为90、180、270的情况
            rotate_angle = int(rotate_angle)
            if rotate_angle == 90:
                self.img_copy = cv2.rotate(self.img_copy, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_angle == 180:
                self.img_copy = cv2.rotate(self.img_copy, cv2.ROTATE_180)
            elif rotate_angle == 270:
                self.img_copy = cv2.rotate(self.img_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # 旋转角度不为90的倍数时，调用cv2.getRotationMatrix2D和cv2.warpAffine函数
            (h, w) = self.img_copy.shape[:2]
            rotate_angle = -rotate_angle # 由逆时针旋转改为顺时针旋转
            M = cv2.getRotationMatrix2D((h/2, w/2), rotate_angle, scale=1)
            self.img_copy = cv2.warpAffine(self.img_copy, M, (w, h))
        self.display_image(self.img_copy, self.scene_2)

    # flip
    def button_35_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to flip.")
            return
        # 从CheckBox中获取是否水平翻转
        is_flip_horizontal = self.ui.checkBox.isChecked()
        # 从CheckBox_2中获取是否垂直翻转
        is_flip_vertical = self.ui.checkBox_2.isChecked()
        # 翻转图像
        if is_flip_horizontal and is_flip_vertical:
            self.img_copy = cv2.flip(self.img_copy, -1)
        elif is_flip_horizontal:
            self.img_copy = cv2.flip(self.img_copy, 1)
        elif is_flip_vertical:
            self.img_copy = cv2.flip(self.img_copy, 0)
        self.display_image(self.img_copy, self.scene_2)

    # Histogram
    def button_4_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to show histogram.")
            return
        # 计算灰度直方图
        if self.camera_opened:
            histogram(self.img_copy, update=True)
        else:
            histogram(self.img_copy, update=False)
        return

    # grayscale
    def button_3_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to grayscale.")
            return
        if len(self.img_copy.shape) > 2:
            self.img_copy = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)
        self.display_image(self.img_copy, self.scene_2)
        
    # Histogram Equalization
    def button_5_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to equalize histogram.")
            return
        if len(self.img_copy.shape) > 2:
            # 如果是彩色图像，先转换为灰度图像
            self.img_copy = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)
        self.img_copy = cv2.equalizeHist(self.img_copy)
        self.display_image(self.img_copy, self.scene_2)

    # Morphological Diff
    def button_6_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to sharpen.")
            return
        self.get_kernel_size()
        k = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.img_copy = cv2.morphologyEx(self.img_copy, cv2.MORPH_GRADIENT, k)
        self.display_image(self.img_copy, self.scene_2)

    # Gaussian Noise
    def button_19_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to add Gaussian noise.")
            return
        self.img_copy = gaussian_noise(self.img_copy)
        self.display_image(self.img_copy, self.scene_2)
        
    # Salt and Pepper Noise
    def button_18_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to add Salt and Pepper noise.")
            return
        self.img_copy = salt_pepper_noise(self.img_copy, prob=0.006)
        self.display_image(self.img_copy, self.scene_2)
        
    # Mean Filter
    def button_13_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to filter.")
            return
        self.get_kernel_size()
        # 均值滤波
        self.img_copy = cv2.blur(self.img_copy, (self.kernel_size, self.kernel_size))
        self.display_image(self.img_copy, self.scene_2)

    # Median Filter
    def button_14_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to filter.")
            return
        self.get_kernel_size()
        # 中值滤波
        self.img_copy = cv2.medianBlur(self.img_copy, self.kernel_size)
        self.display_image(self.img_copy, self.scene_2)
    
    # Gaussian Filter
    def button_16_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to filter.")
            return
        self.get_kernel_size()
        # 高斯滤波
        self.img_copy = cv2.GaussianBlur(self.img_copy, (self.kernel_size, self.kernel_size), 0)
        self.display_image(self.img_copy, self.scene_2)
    
    # Bilateral Filter
    def button_17_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to filter.")
            return
        self.get_kernel_size()
        # 双边滤波
        self.img_copy = cv2.bilateralFilter(self.img_copy, self.kernel_size, 75, 75)
        self.display_image(self.img_copy, self.scene_2)

    # Sobel
    def button_8_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect edge.")
            return
        self.get_kernel_size()
        # Sobel算子
        self.img_copy = cv2.Sobel(self.img_copy, cv2.CV_64F, 1, 1, ksize=self.kernel_size)
        self.img_copy = cv2.convertScaleAbs(self.img_copy)
        # cv2.imshow("Sobel", self.img_copy)
        self.display_image(self.img_copy, self.scene_2)
    
    # Roberts
    def button_9_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect edge.")
            return
        # self.get_kernel_size()
        # Roberts算子
        if len(self.img_copy.shape) > 2:
            b, g, r = cv2.split(self.img_copy)
            b = roberts(b)
            g = roberts(g)
            r = roberts(r)
            # 合并三个通道
            roberts_img = cv2.merge([b, g, r])
            self.img_copy = roberts_img
        else:
            self.img_copy = roberts(self.img_copy)
        self.img_copy = cv2.convertScaleAbs(self.img_copy)
        # cv2.imshow("Roberts", self.img_copy)
        self.display_image(self.img_copy, self.scene_2)

    # Prewitt
    def button_10_clicked(self):
      if self.img_copy is None:
          # QMessageBox.information(self,"Warning","No image to detect edge.")
          return
      # self.get_kernel_size()
      # Prewitt算子
      if len(self.img_copy.shape) > 2:
          b, g, r = cv2.split(self.img_copy)
          b = prewitt(b)
          g = prewitt(g)
          r = prewitt(r)
          # 合并三个通道
          prewitt_img = cv2.merge([b, g, r])
          self.img_copy = prewitt_img
      else:
          self.img_copy = prewitt(self.img_copy)
      self.img_copy = cv2.convertScaleAbs(self.img_copy)
      # cv2.imshow("Prewitt", self.img_copy)
      self.display_image(self.img_copy, self.scene_2)
      
    # Laplacian
    def button_11_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect edge.")
            return
        self.get_kernel_size()
        # Laplacian算子
        self.img_copy = cv2.Laplacian(self.img_copy, cv2.CV_64F, ksize=self.kernel_size)
        self.img_copy = cv2.convertScaleAbs(self.img_copy)
        # cv2.imshow("Laplacian", self.img_copy)
        self.display_image(self.img_copy, self.scene_2)
    
    # Canny
    def button_12_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect edge.")
            return
        self.get_kernel_size()
        # 从spinBox_2获取下阈值，从spinBox_3获取上阈值
        lower_threshold = self.ui.spinBox_2.value()
        upper_threshold = self.ui.spinBox_3.value()
        # Canny算子
        self.img_copy = cv2.Canny(self.img_copy, lower_threshold, upper_threshold, apertureSize=self.kernel_size)
        # cv2.imshow("Canny", self.img_copy)
        self.display_image(self.img_copy, self.scene_2)

    # Adaptive Threshold
    def button_25_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to segment.")
            return
        # 自适应阈值分割
        block_size=11
        constant=5
        if len(self.img_copy.shape) > 2:
            self.img_copy = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)
        self.img_copy = cv2.adaptiveThreshold(self.img_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
        self.display_image(self.img_copy, self.scene_2)

    # Otsu Threshold
    def button_26_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to segment.")
            return
        # Otsu阈值分割
        if len(self.img_copy.shape) > 2:
            self.img_copy = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(self.img_copy, None, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        ret, thresh = cv2.threshold(self.img_copy, None, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.rectangle(thresh, (10, 2), (100,20), (255, 255, 255), -1)
        cv2.putText(thresh, "Otsu threshold: " + str(ret), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        self.img_copy = thresh
        # if self.keep_processing_flag:
        #     pass
        # else:
        #     # 弹窗显示阈值
        #     QMessageBox.information(self,"Information","Otsu threshold: " + str(ret))
        # 显示阈值
        self.display_image(self.img_copy, self.scene_2)

    # Watershed
    def button_27_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to segment.")
            return
        # 分水岭算法最好使用彩色图像
        if len(self.img_copy.shape) == 2:
            self.img_copy = cv2.cvtColor(self.img_copy, cv2.COLOR_GRAY2BGR)
        # Watershed分割
        if self.keep_processing_flag:
            self.img_copy = watershed(self.img_copy)
        else:
            self.img_copy = watershed_color(self.img_copy)
        self.display_image(self.img_copy, self.scene_2)

    # Background Subtractor(MOG2)
    def button_28_clicked(self):
        if self.img_copy is None or not self.camera_opened:
            # QMessageBox.information(self,"Warning","No image to segment.")
            return
        # 背景减除
        self.img_copy = self.bg_subtractor_MOG2.apply(self.img_copy)
        self.display_image(self.img_copy, self.scene_2)
    
    # Background Subtractor(KNN)
    def button_29_clicked(self):
        if self.img_copy is None or not self.camera_opened:
            # QMessageBox.information(self,"Warning","No image to segment.")
            return
        # 背景减除
        self.img_copy = self.bg_subtractor_KNN.apply(self.img_copy)
        self.display_image(self.img_copy, self.scene_2)
    
    # YOLOv5
    def button_30_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect objects.")
            return
        self.img_copy = object_detection.process(self.img_copy, self.yolov5_model)
        self.display_image(self.img_copy, self.scene_2)
    
    # YOLOv8
    def button_31_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect objects.")
            return
        self.img_copy = object_detection.process(self.img_copy, self.yolov8_model)
        self.display_image(self.img_copy, self.scene_2)
    
    # Face Detection(cascade)
    def button_32_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect face.")
            return
        # 加载分类器
        if self.face_cascade.empty(): # 判断分类器是否为空，避免重复加载
            if not self.face_cascade.load(self.face_cascade_path):
                QMessageBox.information(self,"Warning","Failed to load face cascade classifier from " + self.face_cascade_path)
                self.last_button = None
                return
        # 检测人脸
        self.img_copy = face_detection(self.img_copy, self.face_cascade)
        self.display_image(self.img_copy, self.scene_2)

    # SIFT
    def button_33_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect key points.")
            return
        # 检测关键点
        keypoints = self.sift.detect(self.img_copy, None)
        # 绘制关键点
        self.img_copy = cv2.drawKeypoints(self.img_copy, keypoints, None, color=(0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.display_image(self.img_copy, self.scene_2)
    
    # FAST
    def button_20_clicked(self):
        if self.img_copy is None:
            # QMessageBox.information(self,"Warning","No image to detect key points.")
            return
        # 检测关键点
        keypoints = self.fast.detect(self.img_copy, None)
        # 绘制关键点
        self.img_copy = cv2.drawKeypoints(self.img_copy, keypoints, None, color=(0, 255, 0))
        self.display_image(self.img_copy, self.scene_2)

    # update frame
    def update_frame(self):
        ret, self.image = self.cap.read()
        # 当定时器设定时间过短时，会出现cap.read()阻塞的情况。最好的解决方法是把定时器超时时间设定为与相机帧率相当。或者增加如下对self.image是否为空的判断。
        # self.image = self.read_frame()
        # self.image = self.cap.read_latest_frame()
        if not ret: # 读取帧失败，此处判断非常重要
            return
        # self.img_copy = self.image.copy()
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image(self.image, self.scene)
        # self.keep_processing()

    def display_image(self, img, window):
        # Convert image to pixmap and set it to the window
        pixmap = self.__convert2pixmap(img)
        window.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        window.clear()
        window.addPixmap(pixmap)
        return
    
    def __convert2pixmap(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = QImage(img, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        else:
            frame = QImage(img, img.shape[1], img.shape[0], img.shape[1]*1, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(frame)
        return pixmap
    
    def __load_image_path(self):
        # Load the image
        img_path, _ = QFileDialog.getOpenFileName(self, 'Open file', filter="Image files (*.jpg *.jpeg *.bmp *.png) ") #pylint: disable=line-too-long
        if img_path == '':
            QMessageBox.information(self,"Warning","No file selected.")
            return None
        else:
            return img_path
    
    # 从spinbox获取滤波器大小
    def get_kernel_size(self):
        self.kernel_size = self.ui.spinBox.value()
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            QMessageBox.information(self,"Warning","Kernel size should be positive and odd.Set to default value 3.")
            self.kernel_size = 3
            self.ui.spinBox.setValue(3)
            return
        return

    # 检查spinbox的值是否合法
    def check_spinbox_values(self):
        # 如果spinBox_2的值大于spinBox_3的值，则将spinBox_2的值设置为spinBox_3的值减1
        if self.ui.spinBox_2.value() >= self.ui.spinBox_3.value():
            # 弹窗提示
            QMessageBox.information(self,"Warning","Lower threshold should be less than upper threshold.")
            self.ui.spinBox_2.setValue(self.ui.spinBox_3.value() - 1)

    # 持续处理图像
    def keep_processing(self):
        # print("Last button: ", self.last_button)
        if self.keep_processing_flag and (self.last_button is not None):
            # 持续处理图像
            self.start_time = time.time() 
            # print("Keep processing")
            if self.image is None:
                return
            self.img_copy = self.image.copy()
            self.button_func_dict[self.last_button]() # 调用上一次按下的按钮对应的函数
            # self.display_image(self.img_copy, self.scene_2)
            # print("Process Time: ", (time.time() - self.start_time)*1000, "ms") # show processing time, to decide the interval of timer
            print('FPS: ', 1/(time.time() - self.start_time+0.001)) # show FPS
        return

    # 更新按钮状态
    def update_button_state(self, button):
        # 若相机未打开或不需要持续处理图像，则直接返回
        if (not self.camera_opened) or (not self.keep_processing_flag):
            self.last_button = None
            return
        # 若字典中没有当前按钮，则停止持续处理图像
        if button not in self.button_func_dict.keys():
            # self.keep_processing_flag = False
            self.last_button = None
            return
        # 若上一次按下的按钮为None，则开始当前按钮的持续处理
        elif self.last_button is None:
            # self.keep_processing_flag = True
            self.last_button = button
            return
        # 若当前按钮与上一次按下的按钮相同，则停止持续处理图像
        elif button == self.last_button:
            # self.keep_processing_flag = False
            self.last_button = None
            return
        # 若当前按钮与上一次按下的按钮不同，则停止上一次按钮的持续处理，开始当前按钮的持续处理
        elif self.last_button is not None:
            self.keep_processing_flag = False
            self.last_button = button
            self.img_copy = None
            self.keep_processing_flag = True
            return

    def clear_windows(self):
        if self.camera_opened:
            # self.keep_processing_flag = True
            self.last_button = None
        self.img_copy = self.image.copy()
        self.scene_2.clear()
        return
    
    # 还原所有窗口
    def clear_all_windows(self):
        # 清空所有内容
        self.scene.clear()
        self.scene_2.clear()
        # 自适应窗口大小
        # self.ui.graphicsView.fitInView(self.ui.graphicsView.sceneRect(), Qt.KeepAspectRatio) 
        # 重置滚动条位置
        # self.ui.graphicsView.horizontalScrollBar().setValue(0)
        # self.ui.graphicsView.verticalScrollBar().setValue(0)
        # self.ui.graphicsView_2.horizontalScrollBar().setValue(0)
        # self.ui.graphicsView_2.verticalScrollBar().setValue(0)
        # 重置滚动条范围
        # self.ui.graphicsView.horizontalScrollBar().setRange(0,0)
        # self.ui.graphicsView.verticalScrollBar().setRange(0,0)
        # self.ui.graphicsView_2.horizontalScrollBar().setRange(0,0)
        # self.ui.graphicsView_2.verticalScrollBar().setRange(0,0)
        # 重置滚动条长度
        # self.ui.graphicsView.resetTransform()
        # self.ui.graphicsView_2.resetTransform()
        # 关闭滚动条
        # self.ui.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭水平滚动条
        # self.ui.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭垂直滚动条
        # self.ui.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭水平滚动条
        # self.ui.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff) # 关闭垂直滚动条
        return

    # 重写closeEvent函数，关闭相机
    def closeEvent(self, event):
      if self.camera_opened or self.cap is not None:
        self.cap.release()
      event.accept()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    widget = Widget()
    widget.setWindowTitle("Image Process Application")
    widget.show()
    sys.exit(App.exec_())