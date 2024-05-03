import os
import sys
import json
import numpy as np
import soundfile as sf

from utils import load_params

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction, QDoubleValidator
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
    QLabel,
    QFileDialog
)

from pathlib import Path
root_path = str(Path(__file__).parent.parent)

from environment import UASEventRenderer


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.renderer = UASEventRenderer()

        self.setWindowTitle('UAV Sound Propagation')
        layout = QGridLayout()

        # flightpath_image = QLabel()
        # pic = QPixmap('flightpath.png')
        # pic = pic.scaledToWidth(300)
        # flightpath_image.setPixmap(pic)
        # # flightpath_image.resize(90, 90)
        # layout.addWidget(flightpath_image, 0, 1)

        # receiver height
        self.last_height_value = '1.5'
        self.receiver_height_box = QLineEdit(self.last_height_value)
        self.receiver_height_box.setFixedWidth(40)
        self.receiver_height_box.editingFinished.connect(
            self.receiver_height_changed)

        # feature check boxes
        self.direct_checkbox = QCheckBox()
        self.reflection_checkbox = QCheckBox()
        self.atmos_checkbox = QCheckBox()
        self.direct_checkbox.checkStateChanged.connect(self.direct_check)
        self.reflection_checkbox.checkStateChanged.connect(self.reflect_check)
        self.atmos_checkbox.checkStateChanged.connect(self.atmos_check)

        self.direct_checkbox.setChecked(True)
        self.reflection_checkbox.setChecked(True)
        self.atmos_checkbox.setChecked(True)

        # ground material dropdown
        materials = ['Grass', 'Soil', 'Asphalt']
        self.material_dropdown = QComboBox()
        self.material_dropdown.addItems(materials)
        self.material_dropdown.currentIndexChanged.connect(
            self.ground_material_changed)

        # loudspeaker mapping dropdown
        mapping_names = list(json.load(open('mappings/mappings.json')).keys())
        self.mapping_dropdown = QComboBox()
        self.mapping_dropdown.addItems(mapping_names)
        self.mapping_dropdown.setCurrentIndex(1)
        self.mapping_dropdown.currentIndexChanged.connect(self.mapping_changed)

        # labels
        self.filepath_label = QLabel('')
        self.sourcelen_label = QLabel('')
        self.pathlen_label = QLabel('')

        # form
        form = QFormLayout()
        form.addRow('Receiver Height [m]', self.receiver_height_box)
        form.addRow('Direct Path', self.direct_checkbox)
        form.addRow('Ground Reflection', self.reflection_checkbox)
        form.addRow('Atmospheric Absorption', self.atmos_checkbox)
        form.addRow('Ground Material', self.material_dropdown)
        form.addRow('Loudspeaker Mapping', self.mapping_dropdown)
        form.addRow('Source File:', self.filepath_label)
        form.addRow('Source Length:', self.sourcelen_label)
        form.addRow('Path Length:', self.pathlen_label)
        layout.addLayout(form, 0, 0, 1, 1)

        # flightpath table
        # TODO: set what happens when values manually changed
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
        open_flightpath_action = QAction('&Open Flightpath', self)
        open_source_action = QAction('&Open Source Audio', self)
        file_menu = menu.addMenu('&File')
        file_menu.addAction(new_action)
        file_menu.addAction(open_flightpath_action)
        file_menu.addAction(open_source_action)
        open_flightpath_action.triggered.connect(self.open_flightpath)
        open_source_action.triggered.connect(self.open_source)

    def open_source(self, _):
        # set up dialog box
        dialog = QFileDialog()
        # allow single file only
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        # set accepted formats
        dialog.setNameFilter('(*.wav)')
        dialog.exec()
        filepath = dialog.selectedFiles()
        print(filepath)
        if not filepath:
            return False
        filepath = filepath[0]

        self.source, fs = sf.read(filepath)
        # TODO: check source and renderer fs match
        self.filepath_label.setText(os.path.basename(filepath))

        sourcetime_seconds = len(self.source) / fs
        self.sourcelen_label.setText(f'{sourcetime_seconds:.1f} s')

    def direct_check(self):
        if self.direct_checkbox.isChecked():
            self.direct = True
        else:
            self.direct = False
    
    def reflect_check(self):
        if self.reflection_checkbox.isChecked():
            self.reflection = True
        else:
            self.reflection = False

    def atmos_check(self):
        if self.atmos_checkbox.isChecked():
            self.atmos = True
        else:
            self.atmos = False

    def mapping_changed(self, index):
        self.renderer.loudspeaker_mapping = \
            self.mapping_dropdown.itemText(index)
        
    def ground_material_changed(self, index):
        self.renderer.ground_material = self.material_dropdown.itemText(index)

    def receiver_height_changed(self):
        validator = QDoubleValidator(0.0, 20.0, 2)
        value = self.receiver_height_box.text()
        if validator.validate(value, 1)[0].value == 2:
            self.renderer.receiver_height = float(value)
            self.last_height_value = value
        else:
            self.receiver_height_box.setText(self.last_height_value)

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

        table_params = np.loadtxt(
            filepath,
            delimiter=',',
            skiprows=1,
            usecols=np.arange(0, 4),
            dtype='str'
        ).reshape(-1, 4)
        
        renderer_params = load_params(filepath)
        
        self.set_flightpath_table_vals(table_params)
        self.renderer.flight_parameters = renderer_params
        
        pathtime_seconds = len(self.renderer._flightpath.T) / self.renderer.fs
        self.pathlen_label.setText(f'{pathtime_seconds:.1f} s')

    def setup_flightpath_table(self):
        self.flightpath_table = QTableWidget()
        self.flightpath_table.setColumnCount(4)
        self.flightpath_table.setRowCount(1)

        self.flightpath_table.setHorizontalHeaderLabels([
            'Label',
            'Start x y z [m]', 
            'End x y z [m]',
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