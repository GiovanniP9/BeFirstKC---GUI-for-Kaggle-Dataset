# tabs/load_tab.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QSpinBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt

class LoadTab(QWidget):
    """Tab per il caricamento dei dataset, anteprima e selezione del target."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Messaggio iniziale
        label = QLabel("Please load Train dataset and select the target.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Pulsanti di caricamento e selector Dataset
        hlayout = QHBoxLayout()
        btn_train = QPushButton('Load Train Dataset')
        btn_test  = QPushButton('Load Test Dataset')
        btn_train.clicked.connect(lambda: self.parent.load_file('train'))
        btn_test.clicked.connect(lambda: self.parent.load_file('test'))
        hlayout.addWidget(btn_train)
        hlayout.addWidget(btn_test)
        hlayout.addStretch()
        hlayout.addWidget(QLabel('Preview:'))
        combo_ds = QComboBox()
        combo_ds.addItems(['Train', 'Test'])
        combo_ds.currentIndexChanged.connect(self.parent.update_head_view)
        hlayout.addWidget(combo_ds)
        layout.addLayout(hlayout)

        # Selezione Target Column**  
        # Questo combo verrà popolato automaticamente con le colonne di train
        hlayout_tgt = QHBoxLayout()
        hlayout_tgt.addWidget(QLabel('Target Column:'))
        combo_load_target = QComboBox()
        combo_load_target.addItem('Select Target')
        hlayout_tgt.addWidget(combo_load_target)
        layout.addLayout(hlayout_tgt)
        # Quando cambio qui, aggiorno automaticamente ModelTab
        combo_load_target.currentTextChanged.connect(
            lambda txt: self.parent.combo_target.setCurrentText(txt)
        )
        hlayout_tgt.addWidget(combo_load_target)
        layout.addLayout(hlayout_tgt)

        # Numero di righe da mostrare nella preview
        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(QLabel('Rows to show:'))
        spin_rows = QSpinBox()
        spin_rows.setRange(1, 1000)
        spin_rows.setValue(5)
        spin_rows.valueChanged.connect(self.parent.update_head_view)
        hlayout2.addWidget(spin_rows)
        hlayout2.addStretch()
        layout.addLayout(hlayout2)

        # Tabella anteprima
        table_head = QTableWidget()
        table_head.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(table_head)

        # Statistiche numeriche
        layout.addWidget(QLabel('Statistics (numeric columns):'))
        table_desc = QTableWidget()
        table_desc.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(table_desc)

        self.setLayout(layout)

        # Associa widget al parent
        self.parent.combo_head_ds = combo_ds
        self.parent.spin_head_rows = spin_rows
        self.parent.table_head  = table_head
        self.parent.table_describe = table_desc
        self.parent.combo_load_target = combo_load_target
