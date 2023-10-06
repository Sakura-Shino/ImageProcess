import sys
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QApplication
from widget import Widget

if __name__ == "__main__":
    App = QApplication(sys.argv)
    widget = Widget()
    widget.setWindowTitle("Image Process Application")
    widget.show()
    sys.exit(App.exec_())