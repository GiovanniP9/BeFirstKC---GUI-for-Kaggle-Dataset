# tabs/submission_tab.py
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem
)

class SubmissionTab(QWidget):
    """Tab per generare e preview del file di submission."""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Submission'))

        # Selezione colonna ID
        hb = QHBoxLayout()
        hb.addWidget(QLabel('ID column:'))
        combo_id = QComboBox()
        hb.addWidget(combo_id)
        layout.addLayout(hb)

        # Pulsante per generare submission
        gen_btn = QPushButton('Generate Submission')
        gen_btn.clicked.connect(self.parent.generate_submission)
        layout.addWidget(gen_btn)

        # Label e tabella di preview
        layout.addWidget(QLabel('Preview of Submission File (first 15 rows):'))
        table_preview = QTableWidget()
        table_preview.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(table_preview)

        self.setLayout(layout)

        # Associa widget al parent per poterci interagire da DataAnalysisTool
        self.parent.combo_id = combo_id
        self.parent.table_submission_preview = table_preview
