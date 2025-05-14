import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFontDatabase, QFont

from data_analysis_tool import DataAnalysisTool

def main():
    # Punto di ingresso dell'applicazione.
    app = QApplication(sys.argv)

    # Registra il font Roboto
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "Roboto-Regular.ttf")
    font_id = QFontDatabase.addApplicationFont(font_path)
    if font_id != -1:
        family = QFontDatabase.applicationFontFamilies(font_id)[0]
        app.setFont(QFont(family))
    else:
        print("⚠️ Roboto non caricato, uso il font di sistema.")

    # Carica e applica lo stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    with open(qss_path, 'r', encoding='utf-8') as f:
        app.setStyleSheet(f.read())

    # Istanzia e mostra la finestra principale
    window = DataAnalysisTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
