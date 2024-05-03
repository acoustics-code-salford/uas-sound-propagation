import os
import sys
import json
import numpy as np

from utils import load_params

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QApplication, 
    QComboBox,
    QFormLayout,
    QPushButton, 
    QGridLayout, 
    QWidget, 
    QTableWidget,
    QTableWidgetItem,
    QMainWindow,
    QLineEdit,
    QToolBar,
    QLabel,
    QFileDialog
)

from pathlib import Path
root_path = str(Path(__file__).parent.parent)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('UAV Sound Propagation')
        layout = QGridLayout()

        flightpath_image = QLabel()
        pic = QPixmap('flightpath.png')
        pic = pic.scaledToWidth(300)
        flightpath_image.setPixmap(pic)
        # flightpath_image.resize(90, 90)
        layout.addWidget(flightpath_image, 0, 1)

        # receiver height
        self.receiver_height_box = QLineEdit('1.5')
        self.receiver_height_box.setFixedWidth(35)

        # feature check boxes
        self.direct_checkbox = QCheckBox()
        self.reflection_checkbox = QCheckBox()
        self.atmos_checkbox = QCheckBox()

        # ground material dropdown
        materials = ['Grass', 'Soil', 'Asphalt']
        self.material_dropdown = QComboBox()
        self.material_dropdown.addItems(materials)

        # loudspeaker mapping dropdown
        mapping_names = list(json.load(open('mappings/mappings.json')).keys())
        self.mapping_dropdown = QComboBox()
        self.mapping_dropdown.addItems(mapping_names)

        # labels
        filepath_label = QLabel('testscr.wav (19.0 s)')
        pathlen_label = QLabel('19.0 s')

        # form
        form = QFormLayout()
        form.addRow('Receiver Height [metres]', self.receiver_height_box)
        form.addRow('Direct Path', self.direct_checkbox)
        form.addRow('Ground Reflection', self.reflection_checkbox)
        form.addRow('Atmospheric Absorption', self.atmos_checkbox)
        form.addRow('Ground Material', self.material_dropdown)
        form.addRow('Loudspeaker Mapping', self.mapping_dropdown)
        form.addRow('Source File:', filepath_label)
        form.addRow('Path Length:', pathlen_label)
        layout.addLayout(form, 0, 0, 1, 1)

        # flightpath table
        self.setup_flightpath_table()
        layout.addWidget(self.flightpath_table, 1, 0, 1, 2)

        # table control buttons
        self.add_button = QPushButton('+')
        self.add_button.setMaximumWidth(30)
        layout.addWidget(self.add_button, 2, 0)
        self.remove_button = QPushButton('-')
        self.remove_button.setMaximumWidth(30)
        layout.addWidget(self.remove_button, 2, 1)

        self.render_button = QPushButton('Render')
        layout.addWidget(self.render_button, 3, 0, 1, 2)
        
        # set up main display
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        menu = self.menuBar()
        new_action = QAction('&New', self)
        open_action = QAction('&Open Flightpath', self)
        file_menu = menu.addMenu('&File')
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        open_action.triggered.connect(self.open_flightpath)

    def open_flightpath(self, _):
        # set up dialog box
        dialog = QFileDialog()
        # allow single file only
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        # set accepted formats
        dialog.setNameFilter('(*.csv)')
        dialog.exec()
        filepath = dialog.selectedFiles()

        if not filepath:
            return False
        filepath = filepath[0]

        params = np.loadtxt(
            filepath,
            delimiter=',',
            skiprows=1,
            usecols=np.arange(0, 4),
            dtype='str'
        ).reshape(-1, 4)
        #load_params(filepath)
        self.set_flightpath_table_vals(params)

    def setup_flightpath_table(self):
        self.flightpath_table = QTableWidget()
        self.flightpath_table.setColumnCount(4)
        self.flightpath_table.setRowCount(1)

        self.flightpath_table.setHorizontalHeaderLabels([
            'Label',
            'Start [xyz]', 
            'End [xyz]',
            'Speeds [m/s]',
        ])
        
        for col in range(self.flightpath_table.columnCount()):
            self.flightpath_table.setColumnWidth(col, 148)
        self.flightpath_table.setFixedSize(QSize(610, 200))

    def set_flightpath_table_vals(self, params):
        
        # set table to correct number of rows
        n_rows = len(params)
        for i in range(n_rows - self.flightpath_table.rowCount()):
            self.flightpath_table.insertRow(i)
        for i in range(self.flightpath_table.rowCount() - n_rows):
            self.flightpath_table.removeRow(0)

        for row, stage in enumerate(params):
            for col, section, in zip(
                range(self.flightpath_table.columnCount()), stage):
                item = QTableWidgetItem(str(section))
                self.flightpath_table.setItem(row, col, item)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())