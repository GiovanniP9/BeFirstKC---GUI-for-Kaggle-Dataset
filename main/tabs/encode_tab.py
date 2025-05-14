from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QListWidget, QCheckBox, QMessageBox
)

class EncodeTab(QWidget):
    """Tab per l'encoding delle colonne categoriche."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Dataset
        layout.addWidget(QLabel('Dataset:'))
        combo_ds = QComboBox()
        combo_ds.addItems(['Train','Test','Both'])
        combo_ds.currentIndexChanged.connect(self.parent.update_enc)
        layout.addWidget(combo_ds)

        # Select/Deselect
        hb = QHBoxLayout()
        btn_all  = QPushButton('Select All')
        btn_none = QPushButton('Deselect All')
        btn_all.clicked.connect(lambda: list_cols.selectAll())
        btn_none.clicked.connect(lambda: list_cols.clearSelection())
        hb.addWidget(btn_all)
        hb.addWidget(btn_none)
        layout.addLayout(hb)

        # Lista colonne
        list_cols = QListWidget()
        list_cols.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(QLabel('Categorical Columns:'))
        layout.addWidget(list_cols)

        # Opzioni encoding
        cb_label  = QCheckBox('Label Encoding')
        cb_onehot = QCheckBox('OneHot Encoding')
        layout.addWidget(cb_label)
        layout.addWidget(cb_onehot)

        # Pulsante Apply
        apply_btn = QPushButton('Apply Encoding')
        apply_btn.clicked.connect(self.parent.apply_enc)
        layout.addWidget(apply_btn)

        self.setLayout(layout)

        # Associa widget al parent
        self.parent.combo_enc_ds = combo_ds
        self.parent.list_enc     = list_cols
        self.parent.cb_label     = cb_label
        self.parent.cb_onehot    = cb_onehot
