from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QComboBox, QSpinBox, QPlainTextEdit,
    QMessageBox, QFileDialog
)

class ModelTab(QWidget):
    """Tab per definire target, scegliere modello, ottimizzare e addestrare."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Drop columns
        layout.addWidget(QLabel('Drop Columns:'))
        hb = QHBoxLayout()
        btn_all  = QPushButton('Select All')
        btn_none = QPushButton('Deselect All')
        btn_all.clicked.connect(lambda: list_drop.selectAll())
        btn_none.clicked.connect(lambda: list_drop.clearSelection())
        hb.addWidget(btn_all)
        hb.addWidget(btn_none)
        layout.addLayout(hb)
        list_drop = QListWidget()
        list_drop.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(list_drop)
        drop_btn = QPushButton('Drop')
        drop_btn.clicked.connect(self.parent.model_drop)
        layout.addWidget(drop_btn)

        # Target e modello
        layout.addWidget(QLabel('Target:'))
        combo_target = QComboBox()
        layout.addWidget(combo_target)
        layout.addWidget(QLabel('Model:'))
        combo_model = QComboBox()
        combo_model.addItems([
            'XGBoost Classifier','XGBoost Regressor',
            'LightGBM Classifier','LightGBM Regressor'
        ])
        layout.addWidget(combo_model)

        # Train & Optimize
        train_btn = QPushButton('Train')
        train_btn.clicked.connect(self.parent.train_model)
        layout.addWidget(train_btn)
        hb2 = QHBoxLayout()
        hb2.addWidget(QLabel('Trials:'))
        spin_trials = QSpinBox()
        spin_trials.setRange(1,500)
        spin_trials.setValue(50)
        hb2.addWidget(spin_trials)
        opt_btn = QPushButton('Optimize')
        opt_btn.clicked.connect(self.parent.optimize_model)
        hb2.addWidget(opt_btn)
        layout.addLayout(hb2)

        # Train best & Save
        train_best_btn = QPushButton('Train Best Params')
        train_best_btn.setEnabled(False)
        train_best_btn.clicked.connect(self.parent.train_best)
        layout.addWidget(train_best_btn)
        save_btn = QPushButton('Save Model')
        save_btn.setEnabled(False)
        save_btn.clicked.connect(self.parent.save_model)
        layout.addWidget(save_btn)

        # Output risultati
        text_res = QPlainTextEdit()
        text_res.setReadOnly(True)
        layout.addWidget(text_res)

        self.setLayout(layout)

        # Associa widget al parent
        self.parent.list_model_drop = list_drop
        self.parent.combo_target     = combo_target
        self.parent.combo_model      = combo_model
        self.parent.spin_trials      = spin_trials
        self.parent.btn_train_best   = train_best_btn
        self.parent.btn_save_model   = save_btn
        self.parent.text_model_res   = text_res
