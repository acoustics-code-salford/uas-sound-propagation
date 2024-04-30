import os
import sys
import json

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

        # toolbar
        # toolbar = QToolBar('Main Toolbar')
        # self.addToolBar(toolbar)
        # file_action = QAction('File', self)
        # edit_action = QAction('Edit', self)
        # view_action = QAction('View', self)
        # toolbar.addAction(file_action)
        # toolbar.addAction(edit_action)
        # toolbar.addAction(view_action)

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

        flightpath = load_params(filepath)
        print(flightpath)

    def setup_flightpath_table(self):
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
        # maybe get it to load a default flightpath?
        self.set_flightpath_table_vals(init_flightpath)
        self.flightpath_table.setFixedSize(QSize(616, 100))

    def set_flightpath_table_vals(self, vals):
        for col, val in zip(
            range(self.flightpath_table.columnCount()), vals):

            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.flightpath_table.setColumnWidth(col, 75)
            self.flightpath_table.setItem(0, col, item)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())