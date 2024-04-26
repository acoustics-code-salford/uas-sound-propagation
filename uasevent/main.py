import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QPushButton, QGridLayout, QWidget, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        topBtn = QPushButton(parent=self)
        centerBtn = QPushButton(text="Center")
        bottomBtn = QPushButton(text="Bottom")
        layout = QGridLayout()
        layout.addWidget(topBtn, 0, 0, 1, 2)
        layout.addWidget(centerBtn, 1, 0)
        layout.addWidget(bottomBtn, 1, 1)
        
        # set up main display
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())