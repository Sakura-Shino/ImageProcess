from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建QMovie对象
        self.movie = QMovie("./resources/output.gif")
        if self.movie is not None:
            print("movie is not None")

        # 创建QLabel对象
        self.label = QLabel(self)
        self.label.setMovie(self.movie)

        # 设置窗口属性
        self.setCentralWidget(self.label)
        self.setFixedSize(1280,720)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # 启动动画
        self.movie.start()

        # 创建QTimer对象
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.closeAnimation)

        # 启动定时器
        self.timer.start(3000)

    def closeAnimation(self):
        # 停止动画并关闭窗口
        self.movie.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()