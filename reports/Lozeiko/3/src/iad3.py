from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math

# Определяем устройство (GPU, если доступно, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка данных
infrared_thermography_temperature = fetch_ucirepo(id=925)
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets['aveOralM']  # Регрессия по aveOralM

# Предобработка: обработка категориальных фич (one-hot), масштабирование
categorical_cols = ['Gender', 'Age', 'Ethnicity']  # Категориальные
X_encoded = pd.get_dummies(X, columns=categorical_cols, dtype=float)
X = X_encoded.fillna(X_encoded.median())  # Заполнение NaN

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1, 1)

# Разделение на train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Тензоры
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

# Модель MLP
class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes=[128, 64, 32], output_size=1):
        super().__init__()
        layers = []
        in_size = input_size
        for out_size in layer_sizes:
            layers.append(nn.Linear(in_size, out_size))
            in_size = out_size
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(in_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return x

# Функция обучения
def train_model(model, X_train_t, y_train_t, epochs=200, lr=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}')

def evaluate_model(model, X_test_t, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()
    
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    mae = mean_absolute_error(y_test_np, predictions)
    rmse = math.sqrt(mean_squared_error(y_test_np, predictions))
    mape = mean_absolute_percentage_error(y_test_np, predictions) * 100
    
    print(f'MAE: {mae:.4f} °C')
    print(f'RMSE: {rmse:.4f} °C')
    print(f'MAPE: {mape:.4f}%')

# --- Модель без предобучения ---
print('Без предобучения:')
input_size = X.shape[1]
model_without = MLP(input_size).to(device)
train_model(model_without, X_train_t, y_train_t)
evaluate_model(model_without, X_test_t, y_test)

# Автоэнкодер
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(ae, X_t, epochs=50, lr=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    dataset = TensorDataset(X_t, X_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        ae.train()
        epoch_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            out = ae(batch_x)
            loss = criterion(out, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'AE Layer {len(encoders)+1} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}')

# --- Модель с предобучением ---
print('\nПредобучение автоэнкодерами:')
layer_sizes = [128, 64, 32]
current_input = X_train_t
encoders = []

for hidden_size in layer_sizes:
    ae = AutoEncoder(current_input.shape[1], hidden_size).to(device)
    train_autoencoder(ae, current_input)
    encoders.append(ae.encoder)

    # Получаем представления для следующего слоя
    with torch.no_grad():
        current_input = ae.relu(ae.encoder(current_input))

# Создаем предобученную модель
model_with = MLP(input_size, layer_sizes).to(device)

# Копируем веса из обученных энкодеров
for i, encoder_layer in enumerate(encoders):
    model_with.layers[i].load_state_dict(encoder_layer.state_dict())

print('\nДообучение модели с предобученными весами:')
train_model(model_with, X_train_t, y_train_t)

print('\nОценка модели с предобучением:')
evaluate_model(model_with, X_test_t, y_test)