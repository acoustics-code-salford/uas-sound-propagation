import sys

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QApplication, 
    QPushButton, 
    QGridLayout, 
    QWidget, 
    QTableWidget,
    QTableWidgetItem,
    QMainWindow,
    QAbstractScrollArea,
    QHeaderView
)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(QSize(1200, 600))
        layout = QGridLayout()

        self.setup_flighpath_table()
        layout.addWidget(self.flightpath_table)
        
        # set up main display
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def setup_flighpath_table(self):
        self.flightpath_table = QTableWidget()
        self.flightpath_table.setColumnCount(8)
        self.flightpath_table.setRowCount(1)
        self.flightpath_table.setHorizontalHeaderLabels([
            'Start X', 
            'Start Y', 
            'Start Z',
            'End X',
            'End Y',
            'End Z',
            'Start Speed',
            'End Speed'
        ])
        
        # crudely set values and column widths
        init_flightpath = [
            '20', '-142.5', '10', '20', '142.5', '10', '15', '15'
        ]
        for col, val in zip(
            range(self.flightpath_table.columnCount()), init_flightpath):
            
            self.flightpath_table.setColumnWidth(col, 75)
            self.flightpath_table.setItem(0, col, QTableWidgetItem(val))

        self.flightpath_table.setFixedSize(QSize(616, 100))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())