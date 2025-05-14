# tabs/preprocess_tab.py
import io
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSplitter, QPushButton, QListWidget, QGroupBox, QPlainTextEdit
)
from PyQt5.QtCore import Qt

class PreprocessTab(QWidget):
    """Tab per le trasformazioni numeriche e imputazione di NaN."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Selezione dataset
        ds_layout = QHBoxLayout()
        ds_layout.addWidget(QLabel('Dataset:'))
        combo_ds = QComboBox()
        combo_ds.addItems(['Train','Test','Both'])
        combo_ds.currentIndexChanged.connect(self.parent.update_pp_info)
        ds_layout.addWidget(combo_ds)
        layout.addLayout(ds_layout)

        # Splitter per lista colonne e trasformazioni
        splitter = QSplitter(Qt.Horizontal)

        # Colonne selezionabili
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
        left_layout.addWidget(list_cols)
        left.setLayout(left_layout)
        splitter.addWidget(left)

        # Gruppo trasformazioni
        grp = QGroupBox('Transforms')
        grp_layout = QVBoxLayout()
        yj_btn = QPushButton('Yeo-Johnson')
        ss_btn = QPushButton('Standard Scaling')
        yj_btn.clicked.connect(self.parent.apply_yeo)
        ss_btn.clicked.connect(self.parent.apply_std)
        grp_layout.addWidget(yj_btn)
        grp_layout.addWidget(ss_btn)
        grp.setLayout(grp_layout)
        splitter.addWidget(grp)

        layout.addWidget(splitter)

        # Info & Missing
        info_grp = QGroupBox('Info & Missing')
        info_layout = QVBoxLayout()
        txt_info = QPlainTextEdit(); txt_info.setReadOnly(True)
        txt_nan  = QPlainTextEdit(); txt_nan.setReadOnly(True)
        info_layout.addWidget(QLabel('Info:'))
        info_layout.addWidget(txt_info)
        info_layout.addWidget(QLabel('Missing:'))
        info_layout.addWidget(txt_nan)
        info_grp.setLayout(info_layout)
        layout.addWidget(info_grp)

        # Drop e imputazione
        hb = QHBoxLayout()
        drop_btn = QPushButton('Drop NaN')
        drop_btn.clicked.connect(self.parent.pp_dropna)
        combo_imp = QComboBox()
        combo_imp.addItems(['Mean','Median','Mode'])
        imp_btn = QPushButton('Impute Numeric')
        imp_btn.clicked.connect(self.parent.pp_impute)
        hb.addWidget(drop_btn)
        hb.addWidget(combo_imp)
        hb.addWidget(imp_btn)
        layout.addLayout(hb)

        self.setLayout(layout)

        # Associa widget al parent
        self.parent.combo_pp_ds  = combo_ds
        self.parent.list_pp_cols = list_cols
        self.parent.text_info    = txt_info
        self.parent.text_nan     = txt_nan
        self.parent.combo_imp    = combo_imp
