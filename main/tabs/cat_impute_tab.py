import random
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QListWidget, QMessageBox
)

class CatImputeTab(QWidget):
    """Tab per l'imputazione delle colonne categoriche."""
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
        combo_ds.currentIndexChanged.connect(self.parent.update_cat_imp)
        ds_layout.addWidget(combo_ds)
        layout.addLayout(ds_layout)

        # Strategia di imputazione
        strat_layout = QHBoxLayout()
        strat_layout.addWidget(QLabel('Strategy:'))
        combo_strat = QComboBox()
        combo_strat.addItems(['Mode','Constant','Random'])
        strat_layout.addWidget(combo_strat)
        layout.addLayout(strat_layout)

        # Selezione colonne
        btn_all  = QPushButton('Select All')
        btn_none = QPushButton('Deselect All')
        btn_all.clicked.connect(lambda: list_cols.selectAll())
        btn_none.clicked.connect(lambda: list_cols.clearSelection())
        layout.addWidget(btn_all)
        layout.addWidget(btn_none)
        list_cols = QListWidget()
        list_cols.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel('Categorical Columns:'))
        layout.addWidget(list_cols)

        # Pulsante Impute
        imp_btn = QPushButton('Impute')
        imp_btn.clicked.connect(self.parent.apply_cat_impute)
        layout.addWidget(imp_btn)

        self.setLayout(layout)

        # Associa widget al parent
        self.parent.combo_cat_imp_ds       = combo_ds
        self.parent.combo_cat_imp_strategy = combo_strat
        self.parent.list_cat_imp_cols      = list_cols
