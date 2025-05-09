import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping

TRAIN_PATH = '/content/drive/MyDrive/data_bag/train_extra_imputed.csv'
TEST_PATH = '/content/test_cleaned.csv'
NUM_EPOCHS = 30

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

numeric_features = ['Compartments', 'Weight Capacity (kg)']
categorical_features = ['Brand','Material','Size','Laptop Compartment','Waterproof','Style','Color']

numeric_transformer = Pipeline([
    ('yeo', PowerTransformer(method='yeo-johnson')),
    ('std', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X_train = preprocessor.fit_transform(train_data.drop(columns=['id', 'Price']))
y_train = train_data['Price'].values

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

   
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice(f'act_{i}', ['relu', 'tanh', 'elu'])))
        model.add(layers.Dropout(
            hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='mse',
        metrics=['mae'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory='keras_tuner',
    project_name='bag_pricing',
    overwrite=True)

early_stop = EarlyStopping(monitor='val_loss', patience=3)

tuner.search(X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=NUM_EPOCHS,
             callbacks=[early_stop],
             verbose=2)

best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters()[0]

best_model.save('/content/drive/MyDrive/final_submission/best_model.keras')
with open('/content/drive/MyDrive/final_submission/best_params.pkl', 'wb') as f:
    pickle.dump(best_hps, f)

X_test = preprocessor.transform(test_data.drop(columns=['id']))
test_predictions = best_model.predict(X_test).flatten()

output = pd.DataFrame({
    'id': test_data['id'],
    'PredPrice': test_predictions
})
output.to_csv('/content/drive/MyDrive/data_bag/submission_final.csv', index=False)

print("Ottimizzazione completata e file salvati:")
print("- Modello: best_model.keras")
print("- Iperparametri: best_params.pkl")
print("- Predizioni: submission_final.csv")