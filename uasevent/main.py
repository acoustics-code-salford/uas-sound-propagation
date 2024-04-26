import sys
import json

from PyQt6.QtCore import QSize, Qt
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
    QLineEdit
)

from pathlib import Path
root_path = str(Path(__file__).parent.parent)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('UAV Sound Propagation')
        # self.setFixedSize(QSize(650, 600))
        layout = QGridLayout()

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

        form = QFormLayout()
        form.addRow('Receiver Height [metres]', self.receiver_height_box)
        form.addRow('Direct Path', self.direct_checkbox)
        form.addRow('Ground Reflection', self.reflection_checkbox)
        form.addRow('Atmospheric Absorption', self.atmos_checkbox)
        form.addRow('Ground Material', self.material_dropdown)
        form.addRow('Loudspeaker Mapping', self.mapping_dropdown)
        layout.addLayout(form, 0, 0, 1, 2)

        # flightpath table
        self.setup_flighpath_table()
        layout.addWidget(self.flightpath_table, 2, 0, 1, 2)

        # table control buttons
        self.add_button = QPushButton('Add Stage')
        layout.addWidget(self.add_button, 3, 0)
        self.remove_button = QPushButton('Remove Stage')
        layout.addWidget(self.remove_button, 3, 1)
        
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

            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.flightpath_table.setColumnWidth(col, 75)
            self.flightpath_table.setItem(0, col, item)

        self.flightpath_table.setFixedSize(QSize(616, 100))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())