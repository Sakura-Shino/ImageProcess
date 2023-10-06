import cv2
import queue
import threading

# 无缓存读取视频流类
# class VideoCapture:

#     def __init__(self, addr):
#         self.cap = cv2.VideoCapture(addr) # 打开视频流
#         self.q = queue.Queue() # 创建队列
#         self.t = threading.Thread(target=self._reader)
#         self.t.daemon = True
#         self.t.start()

#     # 帧可用时立即读取帧，只保留最新的帧
#     def _reader(self):
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             if not self.q.empty():
#                 try:
#                     self.q.get_nowait()   # 删除上一个（未处理的）帧
#                 except queue.Empty:
#                     pass
#             self.q.put(frame)

#     def read(self):
#         return self.q.get()
    
#     def release(self):
#         self.cap.release()
#         # 停止线程self.t
#         self.t.stop()
#         self.t.join()

class RTSCapture(cv2.VideoCapture):
    """
    Real Time Streaming Capture Class
    """

    def __init__(self, index, apiPreference=cv2.CAP_DSHOW):
        super().__init__(index=index, apiPreference=apiPreference)
        self.q = queue.Queue() # 创建队列
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # 帧可用时立即读取帧，只保留最新的帧
    def _reader(self):
        while True:
            ret, frame = self.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # 删除上一个（未处理的）帧
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read_latest_frame(self):
        return self.q.get()
    
    def release(self):
        super().release()
        # 停止线程self.t
        if self.t.is_alive():
            self.t.join()

if __name__ == '__main__':
    # cap = VideoCapture(0)
    cap = RTSCapture(0, cv2.CAP_DSHOW)
    while True:
        frame = cap.read_latest_frame()
        cv2.imshow("frame", frame)
        if chr(cv2.waitKey(1)&255) == 'q':
            break
