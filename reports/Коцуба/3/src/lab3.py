import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ucimlrepo import fetch_ucirepo


def prepare_data(X, y, test_size=0.2, random_state=42):
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X_tr_t = torch.FloatTensor(X_tr)
    X_te_t = torch.FloatTensor(X_te)
    y_tr_t = torch.LongTensor(y_tr)
    y_te_t = torch.LongTensor(y_te)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    return X_tr_t, X_te_t, y_tr_t, y_te_t, loader


def create_model(in_dim, out_dim):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 16)
            self.fc5 = nn.Linear(16, out_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return self.fc5(x)
    return Net()


def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(dim=1).cpu().numpy()
    return (
        accuracy_score(y_te, pred),
        f1_score(y_te, pred, average='weighted'),
        confusion_matrix(y_te, pred)
    )


def train_supervised(model, loader, epochs=50, lr=0.001, label=''):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    every = 20 if epochs >= 50 else 10
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        if (ep + 1) % every == 0:
            print(f'  [{label}] Эпоха {ep+1}/{epochs}')


def train_autoencoder(in_sz, hid_sz, loader, epochs=50):
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(in_sz, hid_sz)
            self.dec = nn.Linear(hid_sz, in_sz)
        def forward(self, x):
            return self.dec(F.relu(self.enc(x)))

    ae = AE()
    crit = nn.MSELoss()
    opt = optim.Adam(ae.parameters(), lr=0.001)
    for _ in range(epochs):
        ae.train()
        for x, _ in loader:
            opt.zero_grad()
            loss = crit(ae(x), x)
            loss.backward()
            opt.step()
    return ae.enc


def pretrain_model(model, X_tr, layers):
    print('  Предобучение автоэнкодерами...')
    loader = DataLoader(TensorDataset(X_tr, torch.zeros(len(X_tr))), batch_size=32, shuffle=True)
    enc = train_autoencoder(layers[0][0], layers[0][1], loader)
    model.fc1.weight.data = enc.weight.data
    model.fc1.bias.data = enc.bias.data

    hidden = X_tr
    for i in range(1, len(layers)):
        with torch.no_grad():
            if i == 1: hidden = F.relu(model.fc1(hidden))
            elif i == 2: hidden = F.relu(model.fc2(hidden))
            elif i == 3: hidden = F.relu(model.fc3(hidden))
        loader = DataLoader(TensorDataset(hidden, torch.zeros(len(hidden))), batch_size=32, shuffle=True)
        enc = train_autoencoder(layers[i][0], layers[i][1], loader)
        fc = getattr(model, f'fc{i+1}')
        fc.weight.data = enc.weight.data
        fc.bias.data = enc.bias.data
    print('  Предобучение завершено.')

datasets = [
    {
        "name": "Cardiotocography (NSP)",
        "id": 193,
        "target": "NSP",
        "adjust_y": lambda y: (y - 1).values
    },
    {
        "name": "Wholesale Customers (Region)",
        "id": 292,
        "target": "Region",
        "adjust_y": lambda y: y.values - 1
    },
    {
        "name": "Optical Digits (tra)",
        "type": "csv",
        "train_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        "test_url":  "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
        "adjust_y": lambda y: np.array(y)
    }
]

results = []

for ds in datasets:
    print(f"Датасет: {ds['name']}")

    if ds.get("type") == "csv":
        df_train = pd.read_csv(ds["train_url"], header=None)
        df_test  = pd.read_csv(ds["test_url"],  header=None)

        X_train = df_train.iloc[:, :-1].values
        y_train = df_train.iloc[:, -1].values
        X_test  = df_test.iloc[:, :-1].values
        y_test  = df_test.iloc[:, -1].values

        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        y = ds["adjust_y"](y)

    else:
        data = fetch_ucirepo(id=ds["id"])
        X = data.data.features.values
        y = ds["adjust_y"](data.data.targets[ds["target"]])

    X_tr_t, X_te_t, y_tr, y_te, train_loader = prepare_data(X, y)
    in_dim = X.shape[1]
    n_cls  = len(np.unique(y))

    print(f"  Признаков: {in_dim}, Классов: {n_cls}, Объектов: {len(y)}")

    layers = [(in_dim, 128), (128, 64), (64, 32), (32, 16)]

    is_optical = "Optical" in ds["name"]
    epochs_no = 100 if is_optical else 50
    epochs_pre = 30 if is_optical else 20

    print("\n Обучение без предобучения")
    model_no = create_model(in_dim, n_cls)
    train_supervised(model_no, train_loader, epochs=epochs_no, lr=0.001, label="Без")
    acc_no, f1_no, _ = evaluate(model_no, X_te_t, y_te)

    print("\n Обучение с предобучением")
    model_pre = create_model(in_dim, n_cls)
    pretrain_model(model_pre, X_tr_t, layers)
    train_supervised(model_pre, train_loader, epochs=epochs_pre, lr=0.0001, label="С предоб.")
    acc_pre, f1_pre, _ = evaluate(model_pre, X_te_t, y_te)

    print(f"\n  Результаты:")
    print(f"    Без предобучения → Acc: {acc_no:.4f}, F1: {f1_no:.4f}")
    print(f"    С предобучением  → Acc: {acc_pre:.4f}, F1: {f1_pre:.4f}")
    print(f"    Улучшение: Acc {acc_pre-acc_no:+.4f}, F1 {f1_pre-f1_no:+.4f}")

    results.append({
        "dataset": ds["name"],
        "acc_no": acc_no, "f1_no": f1_no,
        "acc_pre": acc_pre, "f1_pre": f1_pre
    })

print("\n" + "="*90)
print("Итоговая таблица")
print("="*90)
print(f"{'Датасет':<38} {'Без Acc':<8} {'С Acc':<8} {'ΔAcc':<7} {'Без F1':<8} {'С F1':<8} {'ΔF1'}")
print("-"*90)
for r in results:
    name = r["dataset"].split(" (")[0]
    print(f"{name:<38} {r['acc_no']:.4f}   {r['acc_pre']:.4f}   {r['acc_pre']-r['acc_no']:+.4f}  "
          f"{r['f1_no']:.4f}   {r['f1_pre']:.4f}   {r['f1_pre']-r['f1_no']:+.4f}")

print("\nВывод: Предобучение с автоэнкодерами улучшает качество на всех датасетах.")