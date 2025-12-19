import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

csv_path = "CASP.csv"

if not os.path.exists(csv_path):
    try:
        from google.colab import files
        print("Файл CASP.csv не найден — выберите файл для загрузки в Colab...")
        uploaded = files.upload()
        first_file = list(uploaded.keys())[0]
        os.rename(first_file, csv_path)
        print(f"Файл загружен и сохранён как {csv_path}")
    except Exception:
        raise FileNotFoundError(f"Файл {csv_path} не найден и не удалось выполнить загрузку через Colab.")
else:
    print(f"Найден локальный файл: {csv_path}")

df = pd.read_csv(csv_path)
print("\nПервые строки датасета:")
display(df.head())
print("\nФорма датасета:", df.shape)
print("\nКолонки:", list(df.columns))

target_col = 'RMSD'
if target_col not in df.columns:
    raise ValueError(f"Целевая переменная '{target_col}' не найдена в файле. Убедитесь, что столбец есть и называется ровно '{target_col}'.")

X = df.drop(columns=[target_col])
y = df[target_col].astype(float)

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

print(f"\nПосле удаления пропусков: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nРазмер train: {X_train_scaled.shape}, Размер test: {X_test_scaled.shape}")
input_dim = X_train_scaled.shape[1]
print(f"Число признаков (input_dim) = {input_dim}")

def create_base_model(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    return model

base_model = create_base_model(input_dim)
base_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='mse',
    metrics=['mae']
)

print("\nАрхитектура базовой модели:")
base_model.summary()

history_base = base_model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
    ]
)

test_loss_base, test_mae_base = base_model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_base = base_model.predict(X_test_scaled)
print(f"\nБазовая модель — Test MSE: {test_loss_base:.4f}, Test MAE: {test_mae_base:.4f}")
print(f"RMSE: {np.sqrt(test_loss_base):.4f}")

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history_base.history['mae'], label='train MAE')
plt.plot(history_base.history['val_mae'], label='val MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.title('MAE')

plt.subplot(1,2,2)
plt.plot(history_base.history['loss'], label='train MSE')
plt.plot(history_base.history['val_loss'], label='val MSE')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.title('Loss (MSE)')
plt.tight_layout()
plt.show()

print("\nЗапускаю предобучение автоэнкодерами (layer-wise)...")

pretrained_weights = []

encoder1 = models.Sequential([layers.Dense(64, activation='relu', input_shape=(input_dim,))])
decoder1 = models.Sequential([layers.Dense(input_dim, activation='linear')])
autoencoder1 = models.Sequential([encoder1, decoder1])
autoencoder1.compile(optimizer='adam', loss='mse')

h1 = autoencoder1.fit(
    X_train_scaled, X_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)
print(f"Autoencoder1 trained. final loss: {h1.history['loss'][-1]:.6f}")
pretrained_weights.append(encoder1.layers[0].get_weights())

encoded1 = encoder1.predict(X_train_scaled)
encoder2 = models.Sequential([layers.Dense(32, activation='relu', input_shape=(64,))])
decoder2 = models.Sequential([layers.Dense(64, activation='linear')])
autoencoder2 = models.Sequential([encoder2, decoder2])
autoencoder2.compile(optimizer='adam', loss='mse')

h2 = autoencoder2.fit(
    encoded1, encoded1,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)
print(f"Autoencoder2 trained. final loss: {h2.history['loss'][-1]:.6f}")
pretrained_weights.append(encoder2.layers[0].get_weights())

encoded2 = encoder2.predict(encoded1)
encoder3 = models.Sequential([layers.Dense(16, activation='relu', input_shape=(32,))])
decoder3 = models.Sequential([layers.Dense(32, activation='linear')])
autoencoder3 = models.Sequential([encoder3, decoder3])
autoencoder3.compile(optimizer='adam', loss='mse')

h3 = autoencoder3.fit(
    encoded2, encoded2,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    verbose=0,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)
print(f"Autoencoder3 trained. final loss: {h3.history['loss'][-1]:.6f}")
pretrained_weights.append(encoder3.layers[0].get_weights())

print("Все автоэнкодеры обучены.")

pretrained_model = models.Sequential()
pretrained_model.add(layers.Dense(64, activation='relu', input_shape=(input_dim,)))
pretrained_model.add(layers.Dropout(0.3))
pretrained_model.add(layers.Dense(32, activation='relu'))
pretrained_model.add(layers.Dropout(0.3))
pretrained_model.add(layers.Dense(16, activation='relu'))
pretrained_model.add(layers.Dropout(0.2))
pretrained_model.add(layers.Dense(1, activation='linear'))

pretrained_model.layers[0].set_weights(pretrained_weights[0])  # layer Dense(64)
pretrained_model.layers[2].set_weights(pretrained_weights[1])  # layer Dense(32)
pretrained_model.layers[4].set_weights(pretrained_weights[2])  # layer Dense(16)

pretrained_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nАрхитектура модели с предобучением:")
pretrained_model.summary()

history_pretrained = pretrained_model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
    ]
)

test_loss_pre, test_mae_pre = pretrained_model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_pre = pretrained_model.predict(X_test_scaled)
print(f"\nМодель с предобучением — Test MSE: {test_loss_pre:.4f}, Test MAE: {test_mae_pre:.4f}")
print(f"RMSE: {np.sqrt(test_loss_pre):.4f}")

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.plot(h1.history['loss'], label='Autoencoder1')
plt.plot(h2.history['loss'], label='Autoencoder2')
plt.plot(h3.history['loss'], label='Autoencoder3')
plt.title('Loss автоэнкодеров')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(2,2,2)
plt.plot(history_base.history['val_mae'], label='Базовая модель val MAE')
plt.plot(history_pretrained.history['val_mae'], label='С предобучением val MAE')
plt.title('Validation MAE')
plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(2,2,3)
plt.plot(history_base.history['val_loss'], label='Базовая модель val MSE')
plt.plot(history_pretrained.history['val_loss'], label='С предобучением val MSE')
plt.title('Validation MSE')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(alpha=0.3)

plt.subplot(2,2,4)
min_epochs = min(len(history_base.history['val_mae']), len(history_pretrained.history['val_mae']), 20)
plt.plot(range(min_epochs), history_base.history['val_mae'][:min_epochs], label='Базовая')
plt.plot(range(min_epochs), history_pretrained.history['val_mae'][:min_epochs], label='С предобучением')
plt.title('Сходимость (первые 20 эпох)')
plt.xlabel('Epoch'); plt.ylabel('Validation MAE'); plt.legend(); plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

mse_base = mean_squared_error(y_test, y_pred_base.flatten())
mae_base = mean_absolute_error(y_test, y_pred_base.flatten())
mse_pre = mean_squared_error(y_test, y_pred_pre.flatten())
mae_pre = mean_absolute_error(y_test, y_pred_pre.flatten())

print("\nИтоговые метрики на тесте:")
print(f"Базовая модель: MSE={mse_base:.4f}, MAE={mae_base:.4f}, RMSE={np.sqrt(mse_base):.4f}")
print(f"С предобучением: MSE={mse_pre:.4f}, MAE={mae_pre:.4f}, RMSE={np.sqrt(mse_pre):.4f}")

save_flag = False
if save_flag:
    pretrained_model.save("pretrained_casp_model.h5")
    print("Сохранено pretrained_casp_model.h5")