from src.visualization import Visualization
import pandas as pd
from src.general_menu import Menu  # Supponendo che tu abbia salvato la classe Menu in menu.py
from src.visualization import Visualization  # Supponendo che tu abbia salvato la classe Visualization


# Carica il dataset
df = pd.read_csv('/home/endershade/Desktop/server_django/data_frame_test.csv')

# Crea un'istanza della classe di visualizzazione
visual = Visualization(df)

# Crea il menu interattivo
menu = Menu(visual)

# Avvia il menu
menu.select_and_execute()