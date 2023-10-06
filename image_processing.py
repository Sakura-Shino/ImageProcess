import numpy as np
import cv2
import time
import matplotlib
matplotlib.use("Qt5Agg") # 使用Qt5Agg后端
import matplotlib.pyplot as plt

def gaussian_noise(image, mean=0, sigma=0.1):
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1) # 裁剪到0-1范围内
    output = np.uint8(output * 255)
    return output


# def salt_pepper_noise(image, prob=0.01): 
#     # output = np.zeros(image.shape, np.uint8)
#     output = np.zeros_like(image, dtype=np.uint8)
#     thres = 1 - prob
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = np.random.rand()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output

# 更高效的实现
def salt_pepper_noise(image, prob=0.01):
    output = np.copy(image)
    h, w = image.shape[:2]
    num_salt = np.ceil(prob * image.size * 0.5)
    salt_coords = np.random.choice(range(h*w), int(num_salt), replace=False)
    salt_coords = np.unravel_index(salt_coords, (h, w))
    output[salt_coords] = 255
    num_pepper = np.ceil(prob * image.size * 0.5)
    pepper_coords = np.random.choice(range(h*w), int(num_pepper), replace=False)
    pepper_coords = np.unravel_index(pepper_coords, (h, w))
    output[pepper_coords] = 0
    return output

def histogram(image, update=False):
    plt.style.use("seaborn-v0_8-whitegrid")
    if update:
        # 清除当前图像，并在原窗口显示新图像
        plt.figure(1, figsize=(10, 8), dpi=100)
        plt.cla()
        plt.clf()
    else:
        plt.figure(figsize=(10, 8), dpi=100)
    if len(image.shape) > 2:
        # 根据不同的颜色通道，计算直方图
        chans = cv2.split(image) # 将图片分割成三个通道
        colors = ("b", "g", "r")
        # plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.title("Flattened Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256]) # 计算直方图
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        # plt.pause(0.001)
        plt.subplots_adjust(hspace=0.5, bottom=0.1)
        plt.draw()
        plt.show() # 显示直方图
  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # 计算灰度图直方图
    hist = cv2.calcHist(image, [0], None, [256], [0, 256])
    # plt.figure(2)
    plt.subplot(2, 1, 2)
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist, color="k")
    plt.xlim([0, 256])
    # plt.pause(0.001) # 暂停0.001s
    plt.subplots_adjust(hspace=0.5, bottom=0.1)
    plt.draw() 
    plt.show() # 显示图像

def roberts(image):
    # 定义Roberts算子
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])

    # 对图像进行滤波
    gradient_x = cv2.filter2D(image, cv2.CV_64F, roberts_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, roberts_y)

    # 计算梯度幅值
    gradient = cv2.magnitude(gradient_x, gradient_y)

    return gradient

def prewitt(image):
    # 定义Prewitt算子
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # 应用Prewitt算子
    gradient_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)

    # 计算总梯度
    gradient = cv2.magnitude(gradient_x, gradient_y)

    # 将梯度值限制在0到255之间
    # 在主程序中采用cv2.convertScaleAbs()函数实现，这里不再重复

    return gradient

def watershed(image):
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对图像进行二值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 对二值化后的图像进行开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 对开运算后的图像进行膨胀，得到背景
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # 对开运算后的图像进行距离变换
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 归一化
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    # 对距离变换的结果进行二值化
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # 对背景图像进行反向处理
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 对前景图像进行标记
    ret, markers = cv2.connectedComponents(sure_fg)
    # 为所有的标记加1，保证背景是0而不是1
    markers = markers + 1
    # 对未知区域进行标记
    markers[unknown == 255] = 0
    # 使用分水岭算法
    markers = cv2.watershed(image, markers)
    # 对标记的结果进行颜色填充
    image[markers == -1] = [255, 0, 0]
    return image

def watershed_color(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological opening
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Create marker labels
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)

    # Generate random colors
    colors = []
    for i in range(1, np.max(markers)+1):
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))

    # Create output image
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Colorize markers
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if markers[i][j] > 1:
                output[i][j] = colors[markers[i][j]-2]
    
    return output

def face_detection(image, face_cascade):
    # 将图像转换为灰度图像
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.equalizeHist(gray)

    # start_time = time.time()
    # 检测人脸
    faces=face_cascade.detectMultiScale(gray)
    # print("Face detection time: {:.4f} ms".format((time.time() - start_time)*1000))
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image, (x,y,w,h),color=(0,0,255),thickness=3)

    return image

import numpy as np
from ultralytics import YOLO

class object_detection():
    
    def __init__(self):
        pass

    @staticmethod
    def process(frame, model):
        results = model(frame, device=0, verbose=False) # 将图像输入到YOLOv8模型中，得到检测结果
        result = results[0] # 取出第一个检测结果（如果有多个结果，表示有多个图像输入）
 
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int") # 将检测结果中的边界框坐标转换为numpy数组，并转换为整数类型
        classes = np.array(result.boxes.cls.cpu(), dtype="int") # 将检测结果中的类别编号转换为numpy数组，并转换为整数类型
        confidences = np.array(result.boxes.conf.cpu(), dtype="float") # 将检测结果中的置信度转换为numpy数组，并转换为浮点数类型
 
        for cls, bbox, confidence in zip(classes, bboxes, confidences):
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
            label = f"{model.names[cls]} {confidence:.2f}"
            # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x, y - label_height - baseline), (x + label_width, y), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        return frame
        
if __name__ == "__main__":
    # model = YOLO('models/yolo/yolov8n.pt') # 加载预训练的 YOLOv8n 模型
    # model(source='./img/physics.jpg', device=0) # 对图像进行预测
    yolov5_model = YOLO('models/yolo/yolov5su.pt') # 加载预训练的 YOLOv5s 模型
    yolov8_model = YOLO('models/yolo/yolov8m.pt') # 加载预训练的 YOLOv8m 模型
    object_detection.process(cv2.imread('./img/physics.jpg'), yolov5_model) # 对图像进行预测
