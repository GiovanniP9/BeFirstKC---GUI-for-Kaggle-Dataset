# tabs/visualize_tab.py
import numpy as np
import seaborn as sns

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QListWidget, QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class VisualizeTab(QWidget):
    """Tab per la visualizzazione dei dati (istogrammi, boxplot, correlazioni)."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Dataset:'))
        combo_ds = QComboBox()
        combo_ds.addItems(['Train','Test'])
        combo_ds.currentIndexChanged.connect(self.parent.update_viz)
        controls.addWidget(combo_ds)

        controls.addWidget(QLabel('Type:'))
        combo_type = QComboBox()
        combo_type.addItems(['Histogram','Boxplot','Correlation'])
        controls.addWidget(combo_type)
        controls.addStretch()
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout()
        btn_all  = QPushButton('Select All')
        btn_none = QPushButton('Deselect All')
        btn_all.clicked.connect(lambda: list_cols.selectAll())
        btn_none.clicked.connect(lambda: list_cols.clearSelection())
        left_layout.addWidget(btn_all)
        left_layout.addWidget(btn_none)
        list_cols = QListWidget()
        list_cols.setSelectionMode(QListWidget.MultiSelection)
        list_cols.itemSelectionChanged.connect(self.parent.plot_viz)
        left_layout.addWidget(list_cols)
        left.setLayout(left_layout)
        splitter.addWidget(left)

        # Canvas per i grafici
        fig = Figure(figsize=self.parent.current_figsize)
        canvas = FigureCanvas(fig)
        area = QScrollArea()
        area.setWidget(canvas)
        area.setWidgetResizable(True)
        splitter.addWidget(area)
        splitter.setSizes([200, 800])

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Associa widget al parent
        self.parent.combo_viz_ds   = combo_ds
        self.parent.combo_viz_type = combo_type
        self.parent.list_viz_cols  = list_cols
        self.parent.figure         = fig
        self.parent.canvas         = canvas
