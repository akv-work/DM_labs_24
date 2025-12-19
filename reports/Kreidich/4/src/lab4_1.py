import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
import os

warnings.filterwarnings('ignore')

file_path = 'CASP.csv'

if os.path.exists(file_path):
    print(f"Загрузка файла из: {file_path}")
    data = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"Файл не найден по пути: {file_path}. Пожалуйста, убедитесь, что имя файла верное.")

y = data[['RMSD']]
X = data.drop(['RMSD'], axis=1)

print("Информация о датасете Protein Tertiary Structure:")
print(f"Размерность признаков (X): {X.shape}")
print(f"Размерность целевой переменной (y): {y.shape}")

print("\nНазвания признаков:")
print(X.columns.tolist())

print("\nНазвания целевых переменных:")
print(y.columns.tolist())

print("\nСтатистика целевой переменной 'RMSD':")
print(y['RMSD'].describe())

print("\nПервые 5 строк признаков:")
print(X.head())

print("\nПервые 5 строк целевой переменной:")
print(y.head())

print(f"\nПропущенные значения в X: {X.isnull().sum().sum()}")
print(f"Пропущенные значения в y: {y.isnull().sum().sum()}")
print("Типы данных в признаках:")
print(X.dtypes)

X_processed = X.copy()

if X_processed.isnull().sum().sum() > 0:
    print("Заполняем пропущенные значения...")
    X_processed = X_processed.fillna(X_processed.mean())

y_target = y['RMSD'].values
y_target_log = np.log1p(y_target)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_target, bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Распределение целевой переменной (RMSD)')
plt.xlabel('RMSD')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_target_log, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Распределение log(1+RMSD)')
plt.xlabel('log(1+RMSD)')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_target_log, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

print(f"\nРазмер train выборки: {X_train_scaled.shape}")
print(f"Размер test выборки: {X_test_scaled.shape}")
print(f"Количество признаков: {X_train_scaled.shape[1]}")

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, momentum=0.9):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.v_bias = np.zeros(n_visible)
        self.h_bias = np.zeros(n_hidden)

        self.W_inc = np.zeros((n_visible, n_hidden))
        self.v_bias_inc = np.zeros(n_visible)
        self.h_bias_inc = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sample_hidden(self, visible):
        activation = np.dot(visible, self.W) + self.h_bias
        p_hidden = self.sigmoid(activation)
        return p_hidden, np.random.binomial(1, p_hidden)

    def sample_visible(self, hidden):
        activation = np.dot(hidden, self.W.T) + self.v_bias
        p_visible = self.sigmoid(activation)
        return p_visible, np.random.binomial(1, p_visible)

    def contrastive_divergence(self, input_data, k=1):
        pos_hidden_probs, pos_hidden_states = self.sample_hidden(input_data)
        hidden_states = pos_hidden_states
        for _ in range(k):
            neg_visible_probs, neg_visible_states = self.sample_visible(hidden_states)
            neg_hidden_probs, neg_hidden_states = self.sample_hidden(neg_visible_states)
            hidden_states = neg_hidden_states

        positive_associations = np.dot(input_data.T, pos_hidden_probs)
        negative_associations = np.dot(neg_visible_states.T, neg_hidden_probs)

        self.W_inc = self.momentum * self.W_inc + \
                    self.learning_rate * ((positive_associations - negative_associations) / len(input_data))
        self.v_bias_inc = self.momentum * self.v_bias_inc + \
                         self.learning_rate * np.mean(input_data - neg_visible_states, axis=0)
        self.h_bias_inc = self.momentum * self.h_bias_inc + \
                         self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

        self.W += self.W_inc
        self.v_bias += self.v_bias_inc
        self.h_bias += self.h_bias_inc

        reconstruction_error = np.mean((input_data - neg_visible_probs) ** 2)
        return reconstruction_error

    def transform(self, data):
        hidden_probs, _ = self.sample_hidden(data)
        return hidden_probs

    def get_weights(self):
        return [self.W, self.h_bias]

X_min = X_train_scaled.min()
X_max = X_train_scaled.max()
X_train_rbm = (X_train_scaled - X_min) / (X_max - X_min)
X_test_rbm = (X_test_scaled - X_min) / (X_max - X_min)

input_dim = X_train_scaled.shape[1]
pretrained_weights_rbm = []

print(f"Размерность входных данных: {input_dim}")

print("\n1. Обучение первого RBM слоя (128 нейронов):")
rbm1 = RBM(n_visible=input_dim, n_hidden=128, learning_rate=0.01, momentum=0.9)

rbm1_errors = []
for epoch in range(50):
    error = rbm1.contrastive_divergence(X_train_rbm, k=1)
    rbm1_errors.append(error)
    if epoch % 10 == 0:
        print(f"   Epoch {epoch}, Reconstruction Error: {error:.4f}")

pretrained_weights_rbm.append(rbm1.get_weights())
hidden1_output = rbm1.transform(X_train_rbm)

print("\n2. Обучение второго RBM слоя (64 нейрона):")
rbm2 = RBM(n_visible=128, n_hidden=64, learning_rate=0.01, momentum=0.9)

rbm2_errors = []
for epoch in range(50):
    error = rbm2.contrastive_divergence(hidden1_output, k=1)
    rbm2_errors.append(error)
    if epoch % 10 == 0:
        print(f"   Epoch {epoch}, Reconstruction Error: {error:.4f}")

pretrained_weights_rbm.append(rbm2.get_weights())
hidden2_output = rbm2.transform(hidden1_output)

print("\n3. Обучение третьего RBM слоя (32 нейрона):")
rbm3 = RBM(n_visible=64, n_hidden=32, learning_rate=0.01, momentum=0.9)

rbm3_errors = []
for epoch in range(50):
    error = rbm3.contrastive_divergence(hidden2_output, k=1)
    rbm3_errors.append(error)
    if epoch % 10 == 0:
        print(f"   Epoch {epoch}, Reconstruction Error: {error:.4f}")

pretrained_weights_rbm.append(rbm3.get_weights())


pretrained_model_rbm = models.Sequential()
pretrained_model_rbm.add(layers.Dense(128, activation='sigmoid', input_shape=(input_dim,), name='pretrained_layer1'))
pretrained_model_rbm.add(layers.Dropout(0.3))
pretrained_model_rbm.add(layers.Dense(64, activation='sigmoid', name='pretrained_layer2'))
pretrained_model_rbm.add(layers.Dropout(0.3))
pretrained_model_rbm.add(layers.Dense(32, activation='sigmoid', name='pretrained_layer3'))
pretrained_model_rbm.add(layers.Dropout(0.2))
pretrained_model_rbm.add(layers.Dense(1, activation='linear', name='output_layer'))

pretrained_model_rbm.layers[0].set_weights([pretrained_weights_rbm[0][0], pretrained_weights_rbm[0][1]])
pretrained_model_rbm.layers[2].set_weights([pretrained_weights_rbm[1][0], pretrained_weights_rbm[1][1]])
pretrained_model_rbm.layers[4].set_weights([pretrained_weights_rbm[2][0], pretrained_weights_rbm[2][1]])

pretrained_model_rbm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

pretrained_model_rbm.summary()

history_pretrained_rbm = pretrained_model_rbm.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ]
)

def create_base_model(input_dim):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    return model

base_model = create_base_model(input_dim)
base_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

base_model.summary()
history_base = base_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ]
)

def evaluate_regression_model(model, X_test, y_test_log, model_name):
    y_pred_log = model.predict(X_test, verbose=0).flatten()
    y_pred = np.expm1(y_pred_log) # Обратное преобразование log1p
    y_true = np.expm1(y_test_log)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return mse, mae, rmse, r2, y_pred

mse_base, mae_base, rmse_base, r2_base, y_pred_base = evaluate_regression_model(
    base_model, X_test_scaled, y_test, "БАЗОВАЯ МОДЕЛЬ")

mse_rbm, mae_rbm, rmse_rbm, r2_rbm, y_pred_rbm = evaluate_regression_model(
    pretrained_model_rbm, X_test_scaled, y_test, "МОДЕЛЬ С RBM")

improvement_mse = ((mse_base - mse_rbm) / mse_base) * 100
improvement_mae = ((mae_base - mae_rbm) / mae_base) * 100
improvement_rmse = ((rmse_base - rmse_rbm) / rmse_base) * 100
improvement_r2 = (r2_rbm - r2_base) * 100

print(f"\nУЛУЧШЕНИЕ RBM МОДЕЛИ ОТНОСИТЕЛЬНО БАЗОВОЙ:")
print(f"MSE: {improvement_mse:+.2f}%")
print(f"MAE: {improvement_mae:+.2f}%")
print(f"RMSE: {improvement_rmse:+.2f}%")
print(f"R²: {improvement_r2:+.2f}%")


plt.figure(figsize=(20, 12))
plt.subplot(2, 3, 1)
metrics = ['MSE', 'MAE', 'RMSE']
base_scores = [mse_base, mae_base, rmse_base]
rbm_scores = [mse_rbm, mae_rbm, rmse_rbm]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, base_scores, width, label='Базовая модель', alpha=0.8)
plt.bar(x + width/2, rbm_scores, width, label='RBM модель', alpha=0.8)

plt.title('Сравнение метрик регрессии', fontsize=14, fontweight='bold')
plt.xlabel('Метрики')
plt.ylabel('Значение')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
models_names = ['Базовая', 'RBM']
r2_scores = [r2_base, r2_rbm]
colors = ['skyblue', 'lightcoral']

bars = plt.bar(models_names, r2_scores, color=colors, alpha=0.8)
plt.title('Сравнение R² Score', fontsize=14, fontweight='bold')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for bar, r2 in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')

plt.subplot(2, 3, 3)
min_epochs = min(len(history_base.history['val_loss']),
                 len(history_pretrained_rbm.history['val_loss']))
epochs_range = range(min_epochs)

plt.plot(epochs_range, history_base.history['val_loss'][:min_epochs],
         label='Базовая модель', linewidth=2)
plt.plot(epochs_range, history_pretrained_rbm.history['val_loss'][:min_epochs],
         label='RBM модель', linewidth=2)
plt.title('Валидационные потери во время обучения', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
plt.plot(rbm1_errors, label='RBM 1 (128 нейронов)', linewidth=2)
plt.plot(rbm2_errors, label='RBM 2 (64 нейрона)', linewidth=2)
plt.plot(rbm3_errors, label='RBM 3 (32 нейрона)', linewidth=2)
plt.title('Ошибки реконструкции RBM', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
y_true = np.expm1(y_test)
plt.scatter(y_true, y_pred_rbm, alpha=0.6, color='red')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
plt.title('RBM модель: Предсказания vs Факт', fontsize=14, fontweight='bold')
plt.xlabel('Фактические значения (RMSD)')
plt.ylabel('Предсказанные значения (RMSD)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.scatter(y_true, y_pred_base, alpha=0.6, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
plt.title('Базовая модель: Предсказания vs Факт', fontsize=14, fontweight='bold')
plt.xlabel('Фактические значения (RMSD)')
plt.ylabel('Предсканные значения (RMSD)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()