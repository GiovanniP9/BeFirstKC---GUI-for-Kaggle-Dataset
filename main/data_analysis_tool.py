import io
import os
import traceback
import pickle
import random

import pandas as pd
import numpy as np
import optuna
import seaborn as sns

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox, QProgressDialog,
    QFileDialog, QTableWidgetItem, QApplication
)
from PyQt5.QtCore import Qt

from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Import dei moduli per ciascuna tab
from tabs.load_tab import LoadTab
from tabs.preprocess_tab import PreprocessTab
from tabs.cat_impute_tab import CatImputeTab
from tabs.encode_tab import EncodeTab
from tabs.visualize_tab import VisualizeTab
from tabs.model_tab import ModelTab
from tabs.submission_tab import SubmissionTab

class DataAnalysisTool(QMainWindow):
    # Finestra principale che gestisce lo stato dei dati e incapsula tutta la logica.
    def __init__(self):
        super().__init__()
        # Attributi di stato
        self.train_df = None
        self.test_df = None
        self.submission_df = None
        self.target_encoder = None
        self.model = None
        self.best_params = None
        self.feature_cols = None
        self.current_figsize = (10, 8)

        # Costruzione UI
        self.init_ui()

    def init_ui(self):
        # Inizializza le tab e configura la finestra.
        self.setWindowTitle('BeFirstKC')
        self.setGeometry(100, 100, 1300, 900)

        # Container di tab
        self.tabs = QTabWidget()
        # Aggiungo tutti i tab importati dai moduli
        self.tabs.addTab(LoadTab(self), 'Load Data')
        self.tabs.addTab(PreprocessTab(self), 'Preprocess')
        self.tabs.addTab(CatImputeTab(self), 'Impute Cat')
        self.tabs.addTab(EncodeTab(self), 'Encode')
        self.tabs.addTab(VisualizeTab(self), 'Visualize')
        self.tabs.addTab(ModelTab(self), 'Model')
        self.tabs.addTab(SubmissionTab(self), 'Submission')

        self.setCentralWidget(self.tabs)

        # Disabilita tutte le tab finché non è selezionato un target valido
        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, i == 0)


        # Abilita le tab quando il target cambia
        self.combo_target.currentIndexChanged.connect(self.check_enable_tabs)
        self.combo_load_target.currentIndexChanged.connect(self.on_load_target_changed)


        self.show()

    def on_load_target_changed(self, idx):
        # Quando cambiamo il combo nel LoadTab, seleziono
        # lo stesso target nel combo_target di ModelTab.
        text = self.combo_load_target.currentText()
        # Trovo l’indice corrispondente in combo_target
        tgt_idx = self.combo_target.findText(text)
        if tgt_idx > 0:  # solo se è una colonna valida
            self.combo_target.setCurrentIndex(tgt_idx)
    
    def check_enable_tabs(self, idx):
        # Abilita tutte le tab se è stato selezionato un target valido.
        if idx > 0:  # >0 significa che è stato selezionato un target reale
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

    def load_file(self, which):
        # Carica train o test dataset tramite file dialog.
        fn, _ = QFileDialog.getOpenFileName(
            self, f'Load {which.title()} Dataset', '',
            'CSV Files (*.csv);;All Files (*)'
        )
        if not fn:
            return
        try:
            df = pd.read_csv(fn)
            if which == 'train':
                self.train_df = df.copy()
            else:
                self.test_df = df.copy()
                self.submission_df = df.copy()
            self.update_all_lists()
            self.update_head_view()
            QMessageBox.information(self, 'Success', f'{which.title()} dataset loaded successfully')
        except Exception as e:
            QMessageBox.critical(self, 'Load Error', str(e))
            traceback.print_exc()

    def update_head_view(self):
        # Aggiorna la tabella di preview e le statistiche numeric.
        ds = self.combo_head_ds.currentText().lower()
        df = getattr(self, f"{ds}_df")
        if df is None:
            self.table_head.clear()
            self.table_describe.clear()
            return

        # Mostra prime n righe
        n = self.spin_head_rows.value()
        head = df.head(n)
        self.table_head.setRowCount(len(head))
        self.table_head.setColumnCount(len(head.columns))
        self.table_head.setHorizontalHeaderLabels(list(head.columns))
        for i, row in enumerate(head.itertuples(index=False)):
            for j, val in enumerate(row):
                self.table_head.setItem(i, j, QTableWidgetItem(str(val)))
        self.table_head.resizeColumnsToContents()

        # Statistiche descrittive colonne numeriche
        desc = df.select_dtypes(include=np.number).describe()
        if desc.empty:
            self.table_describe.clear()
        else:
            self.table_describe.setRowCount(len(desc))
            self.table_describe.setColumnCount(len(desc.columns))
            self.table_describe.setHorizontalHeaderLabels(list(desc.columns))
            self.table_describe.setVerticalHeaderLabels(list(desc.index.astype(str)))
            for i, stat in enumerate(desc.index):
                for j, col in enumerate(desc.columns):
                    self.table_describe.setItem(i, j, QTableWidgetItem(f"{desc.at[stat, col]:.4f}"))
            self.table_describe.resizeColumnsToContents()

    def update_pp_info(self):
        # Aggiorna info e lista colonne nella tab Preprocess.
        ds = self.combo_pp_ds.currentText()
        # Selezione colonne numeriche in base al dataset scelto
        if ds == 'Train':
            df = self.train_df
            numeric_cols = list(df.select_dtypes(include=np.number).columns) if df is not None else []
        elif ds == 'Test':
            df = self.test_df
            numeric_cols = list(df.select_dtypes(include=np.number).columns) if df is not None else []
        else:  # Both: interseco train e test
            if self.train_df is None or self.test_df is None:
                self.list_pp_cols.clear()
                self.text_info.clear()
                self.text_nan.clear()
                return
            cols_train = set(self.train_df.select_dtypes(include=np.number).columns)
            cols_test = set(self.test_df.select_dtypes(include=np.number).columns)
            numeric_cols = sorted(cols_train & cols_test)

        # Rimuovo il target dalla lista delle colonne selezionabili
        tgt = self.combo_target.currentText()
        if tgt in numeric_cols:
            numeric_cols.remove(tgt)

        self.list_pp_cols.clear()
        self.list_pp_cols.addItems(numeric_cols)

        # Mostra info (df.info) e conteggio NaN su train
        if self.train_df is not None:
            buf = io.StringIO()
            self.train_df.info(buf=buf)
            self.text_info.setPlainText(buf.getvalue())
            self.text_nan.setPlainText('\n'.join(f'{c}: {n}' for c, n in self.train_df.isna().sum().items()))
        else:
            self.text_info.clear()
            self.text_nan.clear()

    def apply_yeo(self):
        # Applica Yeo–Johnson alle colonne selezionate.
        ds = self.combo_pp_ds.currentText()
        sel = [i.text() for i in self.list_pp_cols.selectedItems()]
        cols = [c for c in sel if c != self.combo_target.currentText()]
        if not cols:
            QMessageBox.warning(self, 'Transform Error', 'Nessuna colonna valida selezionata')
            return

        # Selezione dei DataFrame
        targets = []
        if ds == 'Both':
            if self.train_df is not None:
                targets.append(('train', self.train_df))
            if self.test_df is not None:
                targets.append(('test', self.test_df))
        else:
            df = self.get_pp_df()
            name = 'train' if ds == 'Train' else 'test'
            targets.append((name, df))

        # Applica Yeo-Johnson dove possibile
        for name, df in targets:
            if df is None:
                continue
            # Rimuovo inf/-inf e NaN
            valid = [c for c in cols if c in df.columns]
            df[valid] = df[valid].replace([np.inf, -np.inf], np.nan).fillna(df[valid].median())
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            for col in valid:
                try:
                    df[[col]] = pt.fit_transform(df[[col]])
                except Exception as e:
                    print(f"[{name}] impossibile trasformare {col}: {e}")

        self.update_pp_info()
        self.update_viz()

    def apply_std(self):
        # Applica StandardScaler alle colonne selezionate.
        ds = self.combo_pp_ds.currentText()
        sel = [i.text() for i in self.list_pp_cols.selectedItems()]
        cols = [c for c in sel if c != self.combo_target.currentText()]
        if not cols:
            QMessageBox.warning(self, 'Transform Error', 'Nessuna colonna valida selezionata')
            return

        if ds == 'Both':
            if self.train_df is not None:
                sc1 = StandardScaler()
                self.train_df[cols] = sc1.fit_transform(self.train_df[cols])
            if self.test_df is not None:
                cols_t = [c for c in cols if c in self.test_df.columns]
                sc2 = StandardScaler()
                self.test_df[cols_t] = sc2.fit_transform(self.test_df[cols_t])
        else:
            df = self.get_pp_df()
            sc = StandardScaler()
            df[cols] = sc.fit_transform(df[cols])

        self.update_pp_info()
        self.update_viz()

    def pp_dropna(self):
        # Rimuove i valori NaN.
        ds = self.combo_pp_ds.currentText()
        if ds == 'Both':
            if self.train_df is not None:
                self.train_df.dropna(inplace=True)
            if self.test_df is not None:
                self.test_df.dropna(inplace=True)
        else:
            df = self.get_pp_df()
            if df is not None:
                df.dropna(inplace=True)
        self.update_pp_info()

    def pp_impute(self):
        # Imputa valori numerici con mean/median/mode.
        ds = self.combo_pp_ds.currentText()
        m = self.combo_imp.currentText()
        targets = []
        if ds in ['Train', 'Both']:
            targets.append(self.train_df)
        if ds in ['Test', 'Both']:
            targets.append(self.test_df)

        for df in targets:
            if df is None:
                continue
            for c in df.select_dtypes(include=np.number).columns:
                if c == self.combo_target.currentText():
                    continue
                if m == 'Mean':
                    df[c] = df[c].fillna(df[c].mean())
                elif m == 'Median':
                    df[c] = df[c].fillna(df[c].median())
                else:
                    df[c] = df[c].fillna(df[c].mode()[0])

        self.update_pp_info()

    def get_pp_df(self):
        # Restituisce il DataFrame di preprocessing attivo.
        return self.train_df if self.combo_pp_ds.currentText() == 'Train' else self.test_df

    def update_cat_imp(self):
        # Aggiorna le colonne categoriali nella tab Impute Cat.
        ds = self.combo_cat_imp_ds.currentText()
        self.list_cat_imp_cols.clear()
        cols = []
        if ds == 'Train':
            if self.train_df is not None:
                cols = self.train_df.select_dtypes(exclude=np.number).columns.tolist()
        elif ds == 'Test':
            if self.test_df is not None:
                cols = self.test_df.select_dtypes(exclude=np.number).columns.tolist()
        else:  # Both
            if self.train_df is not None:
                cols += self.train_df.select_dtypes(exclude=np.number).columns.tolist()
            if self.test_df is not None:
                cols += self.test_df.select_dtypes(exclude=np.number).columns.tolist()
            cols = list(dict.fromkeys(cols))
        self.list_cat_imp_cols.addItems(cols)

    def apply_cat_impute(self):
        # Imputa valori categoriali con modalità, costante o random.
        strategy = self.combo_cat_imp_strategy.currentText()
        cols = [i.text() for i in self.list_cat_imp_cols.selectedItems()]
        if not cols:
            QMessageBox.warning(self, 'Impute Cat', 'No columns selected')
            return
        ds = self.combo_cat_imp_ds.currentText()
        targets = []
        if ds in ['Train', 'Both']:
            targets.append(self.train_df)
        if ds in ['Test', 'Both']:
            targets.append(self.test_df)

        for df in targets:
            if df is None:
                continue
            for c in cols:
                if strategy == 'Mode':
                    df[c].fillna(df[c].mode()[0], inplace=True)
                elif strategy == 'Constant':
                    df[c].fillna('missing', inplace=True)
                else:  # Random
                    vals = df[c].dropna().tolist()
                    df[c] = df[c].apply(lambda x: random.choice(vals) if pd.isna(x) and vals else x)

        QMessageBox.information(self, 'Impute Cat', f'Imputed {cols} using {strategy}')
        self.update_pp_info()
        self.update_enc()
        self.update_viz()
        self._lists()

    def update_enc(self):
        # Aggiorna le colonne categoriali nella tab Encode.
        ds = self.combo_enc_ds.currentText()
        self.list_enc.clear()
        cols = []
        if ds == 'Train':
            if self.train_df is not None:
                cols = self.train_df.select_dtypes(exclude=np.number).columns.tolist()
        elif ds == 'Test':
            if self.test_df is not None:
                cols = self.test_df.select_dtypes(exclude=np.number).columns.tolist()
        else:  # Both
            if self.train_df is not None:
                cols += self.train_df.select_dtypes(exclude=np.number).columns.tolist()
            if self.test_df is not None:
                cols += self.test_df.select_dtypes(exclude=np.number).columns.tolist()
            cols = list(dict.fromkeys(cols))
        self.list_enc.addItems(cols)

    def apply_enc(self):
        # Applica Label e OneHot encoding alle colonne selezionate.
        cols = [i.text() for i in self.list_enc.selectedItems()]
        if not cols:
            QMessageBox.warning(self, 'Encode', 'No columns selected')
            return
        ds = self.combo_enc_ds.currentText()
        datasets = []
        if ds in ['Train', 'Both']:
            datasets.append(('train', self.train_df))
        if ds in ['Test', 'Both']:
            datasets.append(('test', self.test_df))

        for name, df in datasets:
            if df is None:
                continue
            if self.cb_label.isChecked():
                le = LabelEncoder()
                for c in cols:
                    if c in df.columns:
                        df[c] = le.fit_transform(df[c].astype(str))
            if self.cb_onehot.isChecked():
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                exist = [c for c in cols if c in df.columns]
                if exist:
                    arr = ohe.fit_transform(df[exist].astype(str))
                    new_cols = ohe.get_feature_names_out(exist)
                    df_ohe = pd.DataFrame(arr, columns=new_cols, index=df.index)
                    df.drop(columns=exist, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    df = pd.concat([df, df_ohe.reset_index(drop=True)], axis=1)
                    setattr(self, f"{name}_df", df)

        QMessageBox.information(self, 'Encode', 'Applied encoding')
        self.update_enc()
        self.update_pp_info()
        self.update_all_lists()

    def update_viz(self):
        # Aggiorna la lista delle colonne numeriche per la tab Visualize.
        df = self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        self.list_viz_cols.clear()
        if df is not None:
            self.list_viz_cols.addItems(df.select_dtypes(include=np.number).columns)

    def plot_viz(self):
        # Genera il grafico selezionato (Histogram, Boxplot, Correlation).
        df = self.train_df if self.combo_viz_ds.currentText()=='Train' else self.test_df
        cols = [i.text() for i in self.list_viz_cols.selectedItems()]
        if df is None or not cols:
            return
        numdf = df[cols].select_dtypes(include=np.number)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        vt = self.combo_viz_type.currentText()
        try:
            if vt == 'Histogram':
                for c in cols:
                    ax.hist(numdf[c].dropna(), bins=10, alpha=0.5, label=c)
                ax.legend()
            elif vt == 'Boxplot':
                ax.boxplot([numdf[c].dropna() for c in cols], labels=cols)
            else:  # Correlation
                corr = numdf.corr()
                sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool)), ax=ax, cmap='viridis')
                ax.tick_params(axis='both', labelsize=max(6,12-len(cols)))
        except Exception as e:
            QMessageBox.warning(self, 'Plot Error', str(e))

        self.canvas.draw()

    def update_all_lists(self):
        # Aggiorna tutte le liste di colonne e dropdown (preprocess/model/submission).
        cols = list(self.train_df.columns) if self.train_df is not None else []
        # Preprocess & drop
        self.list_pp_cols.clear()
        self.list_model_drop.clear()
        for w in [self.list_pp_cols, self.list_model_drop]:
            w.addItems(cols)
        # Target
        self.combo_target.clear()
        self.combo_target.addItem('Select Target')
        self.combo_target.addItems(cols)
        # Sincronizzo anche il selector in LoadTab
        if hasattr(self, 'combo_load_target'):
            self.combo_load_target.clear()
            self.combo_load_target.addItems(cols)
        if hasattr(self, 'combo_load_target'):
           # blocco i segnali per non triggerare on_load_target_changed mentre popolo
           self.combo_load_target.blockSignals(True)
           self.combo_load_target.clear()
           # 1) placeholder
           self.combo_load_target.addItem('Select Target')
           # 2) vere colonne
           self.combo_load_target.addItems(cols)
           # 3) rimetti indice su 0
           self.combo_load_target.setCurrentIndex(0)
           self.combo_load_target.blockSignals(False)    
           
        # ID per submission
        ids = list(self.submission_df.columns) if self.submission_df is not None else cols
        self.combo_id.clear()
        self.combo_id.addItems(ids)
        # ID per submission
        ids = list(self.submission_df.columns) if self.submission_df is not None else cols
        self.combo_id.clear()
        self.combo_id.addItems(ids)
        # Aggiorna anche le altre liste
        self.update_pp_info()
        self.update_enc()
        self.update_viz()

    def model_drop(self):
        # Rimuove le colonne selezionate prima dell'addestramento.
        to_drop = [i.text() for i in self.list_model_drop.selectedItems()]
        if not to_drop:
            QMessageBox.warning(self,'Model','No columns selected')
            return
        if self.train_df is not None:
            self.train_df.drop(columns=to_drop, inplace=True, errors='ignore')
        if self.test_df is not None:
            self.test_df.drop(columns=to_drop, inplace=True, errors='ignore')
        self.update_all_lists()
        QMessageBox.information(self,'Model',f'Dropped {to_drop}')

    def train_model(self):
        # Addestra il modello scelto con i dati di train.
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first')
            return
        df = self.train_df.copy()
        tgt = self.combo_target.currentText()
        if tgt == 'Select Target' or tgt not in df.columns:
            QMessageBox.warning(self,'Model','Select a valid target')
            return

        X = df.drop(columns=[tgt]).select_dtypes(include=np.number)
        y = df[tgt]
        is_clf = 'Classifier' in self.combo_model.currentText()

        # Encoding automatico del target per classificazione
        if is_clf:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y.astype(str))
            mdl = XGBClassifier(eval_metric='mlogloss') if 'XGBoost' in self.combo_model.currentText() else LGBMClassifier()
        else:
            base = XGBRegressor() if 'XGBoost' in self.combo_model.currentText() else LGBMRegressor()
            mdl = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())

        self.model = mdl
        self.feature_cols = X.columns.tolist()

        # Split train/validation
        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        s = ""
        try:
            self.model.fit(Xtr, ytr)
            # Predizioni
            ytr_pred = self.model.predict(Xtr)
            yv_pred = self.model.predict(Xv)

            if is_clf:
                s += (
                    f"Accuracy: {accuracy_score(yv, yv_pred):.4f}\n"
                    f"Precision: {precision_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"Recall: {recall_score(yv, yv_pred, average='weighted'):.4f}\n"
                )
                cm = confusion_matrix(yv, yv_pred)
                s += f"Confusion Matrix:\n{cm}\n"
            else:
                mae = mean_absolute_error(yv, yv_pred)
                mse = mean_squared_error(yv, yv_pred)
                rmse = np.sqrt(mse)
                s += (
                    f"Train R2: {self.model.score(Xtr, ytr):.4f}\n"
                    f"Val R2: {self.model.score(Xv, yv):.4f}\n"
                    f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n"
                )

            cv = cross_val_score(self.model, Xtr, ytr, cv=5)
            s += f"CV mean: {cv.mean():.4f} (std {cv.std():.4f})"
            self.text_model_res.setPlainText(s)
            self.btn_save_model.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,'Train Error',str(e))
            traceback.print_exc()

    def optimize_model(self):
        # Ottimizza iperparametri via Optuna.
        if self.train_df is None:
            QMessageBox.warning(self,'Model','Load train data first')
            return
        df = self.train_df.copy()
        tgt = self.combo_target.currentText()
        if tgt == 'Select Target' or tgt not in df.columns:
            QMessageBox.warning(self,'Model','Select a valid target')
            return

        X = df.drop(columns=[tgt]).select_dtypes(include=np.number)
        y = df[tgt]
        if 'Classifier' in self.combo_model.currentText() and self.target_encoder:
            y = self.target_encoder.transform(y.astype(str))
        self.feature_cols = X.columns.tolist()

        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        trials = self.spin_trials.value()
        mname = self.combo_model.currentText()
        progress = QProgressDialog('Optuna...', 'Cancel', 0, trials, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            if 'XGBoost' in mname:
                mdl = XGBClassifier(**params, eval_metric='mlogloss') if 'Classifier' in mname else XGBRegressor(**params)
            else:
                mdl = LGBMClassifier(**params) if 'Classifier' in mname else LGBMRegressor(**params)
            score = cross_val_score(mdl, Xtr, ytr, cv=3).mean()
            progress.setValue(trial.number+1)
            QApplication.processEvents()
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trials)
        progress.reset()
        self.best_params = study.best_params
        self.btn_train_best.setEnabled(True)
        self.text_model_res.setPlainText(f"Best params: {self.best_params}\nBest score: {study.best_value:.4f}")

    def train_best(self):
        # Addestra il modello con i migliori parametri trovati.
        if self.train_df is None or not self.best_params:
            return
        is_clf = 'Classifier' in self.combo_model.currentText()
        params = self.best_params
        if is_clf:
            mdl = XGBClassifier(**params, eval_metric='mlogloss') if 'XGBoost' in self.combo_model.currentText() else LGBMClassifier(**params)
        else:
            base = XGBRegressor(**params) if 'XGBoost' in self.combo_model.currentText() else LGBMRegressor(**params)
            mdl = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
        self.model = mdl

        df = self.train_df.copy()
        tgt = self.combo_target.currentText()
        X = df[self.feature_cols]
        y = df[tgt]
        if is_clf:
            y = self.target_encoder.transform(y.astype(str))

        Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
        dlg = QProgressDialog('Training best...', None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        QApplication.processEvents()
        try:
            self.model.fit(Xtr, ytr)
            yv_pred = self.model.predict(Xv)
            s = ""
            if is_clf:
                s += (
                    f"Accuracy: {accuracy_score(yv, yv_pred):.4f}\n"
                    f"Precision: {precision_score(yv, yv_pred, average='weighted'):.4f}\n"
                    f"Recall: {recall_score(yv, yv_pred, average='weighted'):.4f}\n"
                )
                cm = confusion_matrix(yv, yv_pred)
                s += f"Confusion Matrix:\n{cm}\n"
            else:
                mae = mean_absolute_error(yv, yv_pred)
                mse = mean_squared_error(yv, yv_pred)
                rmse = np.sqrt(mse)
                s += (
                    f"Train R2: {self.model.score(Xtr, ytr):.4f}\n"
                    f"Val R2: {self.model.score(Xv, yv_pred):.4f}\n"
                    f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\n"
                )
            cv = cross_val_score(self.model, Xtr, ytr, cv=5)
            s += f"CV mean: {cv.mean():.4f} (std {cv.std():.4f})"
            self.text_model_res.setPlainText(s)
            self.btn_save_model.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,'Train Best Error',str(e))
            traceback.print_exc()

    def save_model(self):
        # Salva il modello in un file .pkl.
        if not self.model:
            QMessageBox.warning(self,'Save Model','No model to save')
            return
        fn, _ = QFileDialog.getSaveFileName(self,'Save Model','','Pickle Files (*.pkl);;All Files (*)')
        if fn:
            try:
                with open(fn,'wb') as f:
                    pickle.dump(self.model, f)
                QMessageBox.information(self,'Save Model',f'Model saved to {fn}')
            except Exception as e:
                QMessageBox.critical(self,'Save Error',str(e))

    def generate_submission(self):
        # Genera il file di submission CSV usando il modello addestrato.
        if not self.model or self.test_df is None:
            QMessageBox.warning(self,'Submission','Train model and load test data first')
            return
        idc = self.combo_id.currentText()
        df = self.test_df.copy()
        subdf = self.submission_df.copy()
        if idc not in subdf.columns:
            QMessageBox.warning(self,'Submission','Invalid ID column')
            return
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            QMessageBox.warning(self,'Submission',f'Missing columns: {missing}')
            return

        X = df[self.feature_cols]
        preds = self.model.predict(X)
        if self.target_encoder:
            try:
                preds = self.target_encoder.inverse_transform(preds)
            except:
                pass

        
        sub = pd.DataFrame({idc: self.submission_df[idc], 'prediction': preds})

        # ——— Popolo la preview nella SubmissionTab ———
        head = sub.head(15)
        tbl = self.table_submission_preview
        tbl.clear()
        tbl.setRowCount(len(head))
        tbl.setColumnCount(len(head.columns))
        tbl.setHorizontalHeaderLabels(list(head.columns))
        for i, row in enumerate(head.itertuples(index=False)):
            for j, val in enumerate(row):
                tbl.setItem(i, j, QTableWidgetItem(str(val)))
        tbl.resizeColumnsToContents()
        
        fn, _ = QFileDialog.getSaveFileName(self,'Save Submission','submission.csv','CSV Files (*.csv)')
        if fn:
            try:
                sub.to_csv(fn, index=False)
                QMessageBox.information(self,'Submission',f'Saved {fn}')
            except Exception as e:
                QMessageBox.critical(self,'Submission Error',str(e))
