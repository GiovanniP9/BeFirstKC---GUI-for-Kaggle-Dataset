import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# file paths
train_path = 'csv\\train.csv'
test_path = 'csv\\test.csv'

# read data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# copia per modifiche
data = train_data.copy()

# colonne con missing values da riempire
cols_to_fill = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof"]

# encoding placeholder per categoriche
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna("missing")

# encoding label delle colonne categoriche
encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Imputazione dei missing values nel dataset originale
for col in cols_to_fill:
    if train_data[col].isnull().sum() == 0:
        continue  # skip se non ci sono valori mancanti

    print(f"Filling missing values for {col}...")

    y = train_data[col]
    X = data.drop(columns=[col])

    X_train = X[y.notnull()]
    y_train = y[y.notnull()]
    X_test = X[y.isnull()]

    if y.dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        y_le = LabelEncoder()
        y_train_encoded = y_le.fit_transform(y_train.astype(str))
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        train_data.loc[y.isnull(), col] = y_le.inverse_transform(y_pred)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_data.loc[y.isnull(), col] = y_pred

print("Missing values filled.")
print("Check nulls:\n", train_data.isnull().sum())

# Selezione target e features
y = train_data['Price']
features = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof"]

X = train_data[features]
X_test = test_data[features]

# Rimuove righe con valori mancanti nel test
X_test_cleaned = X_test.dropna()
test_ids = test_data.loc[X_test_cleaned.index, 'id']

# Unione feature train/test per uniformare l’encoding one-hot
X_all = pd.concat([X, X_test_cleaned], keys=['train', 'test'])
X_all_encoded = pd.get_dummies(X_all)

# Separazione
X_encoded = X_all_encoded.xs('train')
X_test_encoded = X_all_encoded.xs('test')

# split per validazione
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, random_state=42)

# modello finale
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# valutazione
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = np.sqrt(mean_squared_error(y_val, preds))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# predizioni sul test set
test_preds = model.predict(X_test_encoded)

# salvataggio submission
submission = pd.DataFrame({'id': test_ids, 'Price': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")
