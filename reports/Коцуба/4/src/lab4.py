import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
figsize = (10, 6)

def prepare_data(X, y, test_size=0.2, random_state=42, normalize_to_01=False):
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if normalize_to_01:
        X_tr = (X_tr - X_tr.min()) / (X_tr.max() - X_tr.min() + 1e-8)
        X_te = (X_te - X_te.min()) / (X_te.max() - X_te.min() + 1e-8)

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

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        wx = torch.mm(v, self.W.t())
        activation = wx + self.h_bias
        p_h = torch.sigmoid(activation)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        wx = torch.mm(h, self.W)
        activation = wx + self.v_bias
        p_v = torch.sigmoid(activation)
        return p_v, torch.bernoulli(p_v)


def train_rbm(rbm, data_loader, epochs=50, lr=0.01, k=1):
    optimizer = optim.Adam(rbm.parameters(), lr=lr)
    rbm.train()
    losses = []
    for ep in range(epochs):
        epoch_loss = 0.0
        for x, _ in data_loader:
            x = x.clone()
            batch_size = x.size(0)

            v = x
            h = torch.bernoulli(torch.sigmoid(torch.mm(v, rbm.W.t()) + rbm.h_bias))
            for _ in range(k):
                v = torch.bernoulli(torch.sigmoid(torch.mm(h, rbm.W) + rbm.v_bias))
                h = torch.bernoulli(torch.sigmoid(torch.mm(v, rbm.W.t()) + rbm.h_bias))

            pos_h = torch.sigmoid(torch.mm(x, rbm.W.t()) + rbm.h_bias)
            neg_h = torch.sigmoid(torch.mm(v, rbm.W.t()) + rbm.h_bias)

            dW = (torch.mm(x.t(), pos_h) - torch.mm(v.t(), neg_h)) / batch_size
            dv = (x - v).sum(0) / batch_size
            dh = (pos_h - neg_h).sum(0) / batch_size

            optimizer.zero_grad()
            rbm.W.grad = -dW.t()
            rbm.v_bias.grad = -dv
            rbm.h_bias.grad = -dh
            optimizer.step()

            recon_loss = F.mse_loss(v, x)
            epoch_loss += recon_loss.item()

        avg_loss = epoch_loss / len(data_loader)
        losses.append(avg_loss)
        if (ep + 1) % 20 == 0:
            print(f'    RBM Эпоха {ep+1}/{epochs}, Recon Loss: {avg_loss:.6f}')
    return losses

def pretrain_with_rbm(model, X_tr, layers, rbm_epochs=20):
    print('  Предобучение с помощью RBM...')
    input_data = X_tr.clone()
    all_rbm_losses = []

    for i, (in_sz, hid_sz) in enumerate(layers):
        print(f'  Обучение RBM {i+1}: {in_sz} → {hid_sz}')
        rbm = RBM(n_visible=in_sz, n_hidden=hid_sz)
        loader = DataLoader(TensorDataset(input_data, torch.zeros(len(input_data))), batch_size=32, shuffle=True)
        losses = train_rbm(rbm, loader, epochs=rbm_epochs, lr=0.01, k=1)
        all_rbm_losses.append(losses)

        fc = getattr(model, f'fc{i+1}')
        with torch.no_grad():
            fc.weight.data = rbm.W.clone()
            fc.bias.data   = rbm.h_bias.clone()

        with torch.no_grad():
            p_h, _ = rbm.sample_h(input_data)
            input_data = p_h

    print('  Предобучение RBM завершено.')
    return all_rbm_losses

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
    model.fc1.bias.data   = enc.bias.data

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
        fc.bias.data   = enc.bias.data
    print('  Предобучение завершено.')

def train_supervised(model, loader, epochs=50, lr=0.001, label=''):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    every = 20 if epochs >= 50 else 10
    model.train()
    losses = []
    for ep in range(epochs):
        epoch_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            output = model(x)
            loss = crit(output, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (ep + 1) % every == 0:
            print(f'  [{label}] Эпоха {ep+1}/{epochs}, Loss: {avg_loss:.6f}')
    return losses

def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(X_te).argmax(dim=1).cpu().numpy()
    y_true = y_te.cpu().numpy()
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average='weighted')
    cm = confusion_matrix(y_true, pred)
    return acc, f1, cm

def plot_rbm_losses(all_rbm_losses, layers):
    plt.figure(figsize=figsize)
    for i, losses in enumerate(all_rbm_losses):
        plt.plot(losses, label=f'RBM {i+1}: {layers[i][0]}→{layers[i][1]}')
    plt.title('RBM Reconstruction Loss по слоям')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_supervised_losses(losses_dict):
    plt.figure(figsize=figsize)
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label, marker='o', markevery=5)
    plt.title('Cross-Entropy Loss во время финального обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results):
    methods = list(results.keys())
    accs = [r[0] for r in results.values()]
    f1s = [r[1] for r in results.values()]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x + width/2, f1s, width, label='F1-score', color='lightcoral')

    ax.set_ylabel('Score')
    ax.set_title('Сравнение Accuracy и F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0.8, 1.0)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cms, labels, class_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (cm, label) in enumerate(zip(cms, labels)):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(f'Confusion Matrix: {label}')
        axes[i].set_xlabel('Предсказано')
        axes[i].set_ylabel('Истинно')
    plt.tight_layout()
    plt.show()

def run_experiment(dataset_name, dataset_id, target_col=None, class_names=None, normalize_to_01=False):
    print(f"\n{'='*20} {dataset_name} {'='*20}")

    data = fetch_ucirepo(id=dataset_id)
    X = data.data.features.values
    if target_col:
        y = (data.data.targets[target_col] - 1).values if dataset_id == 193 else data.data.targets.values.flatten()
    else:
        y = data.data.targets.values.flatten()

    X_tr_t, X_te_t, y_tr_t, y_te_t, train_loader = prepare_data(X, y, normalize_to_01=normalize_to_01)
    in_dim = X.shape[1]
    n_cls = len(np.unique(y))
    print(f"  Признаков: {in_dim}, Классов: {n_cls}, Объектов: {len(y)}")

    layers = [(in_dim, 128), (128, 64), (64, 32), (32, 16)]
    epochs_no = 100
    epochs_pre = 30
    rbm_epochs = 20

    results = {}
    cms = {}
    supervised_losses = {}

    print("\n1. Обучение без предобучения")
    model_no = create_model(in_dim, n_cls)
    loss_no = train_supervised(model_no, train_loader, epochs=epochs_no, lr=0.001, label="Без")
    acc_no, f1_no, cm_no = evaluate(model_no, X_te_t, y_te_t)
    results["Без"] = (acc_no, f1_no)
    cms["Без"] = cm_no
    supervised_losses["Без"] = loss_no

    print("\n2. Обучение с предобучением (автоэнкодеры)")
    model_ae = create_model(in_dim, n_cls)
    pretrain_model(model_ae, X_tr_t, layers)
    loss_ae = train_supervised(model_ae, train_loader, epochs=epochs_pre, lr=0.001, label="AE")
    acc_ae, f1_ae, cm_ae = evaluate(model_ae, X_te_t, y_te_t)
    results["AE"] = (acc_ae, f1_ae)
    cms["AE"] = cm_ae
    supervised_losses["AE"] = loss_ae

    print("\n3. Обучение с предобучением (RBM)")
    model_rbm = create_model(in_dim, n_cls)
    rbm_losses = pretrain_with_rbm(model_rbm, X_tr_t, layers, rbm_epochs=rbm_epochs)
    loss_rbm = train_supervised(model_rbm, train_loader, epochs=epochs_pre, lr=0.001, label="RBM")
    acc_rbm, f1_rbm, cm_rbm = evaluate(model_rbm, X_te_t, y_te_t)
    results["RBM"] = (acc_rbm, f1_rbm)
    cms["RBM"] = cm_rbm
    supervised_losses["RBM"] = loss_rbm

    print("\n" + "="*80)
    print(f"Итоговая таблица: {dataset_name}")
    print("="*80)
    print(f"{'Метод':<25} {'Accuracy':<10} {'F1-score':<10} {'ΔAcc':<8} {'ΔF1'}")
    print("-"*80)
    base_acc, base_f1 = results["Без"]
    print(f"{'Без предобучения':<25} {base_acc:.4f}     {base_f1:.4f}     {'—':<8} {'—'}")
    for method in ["AE", "RBM"]:
        acc, f1 = results[method]
        print(f"{'+ ' + method:<24} {acc:.4f}     {f1:.4f}     {acc-base_acc:+.4f}   {f1-base_f1:+.4f}")
    print("="*80)

    plot_rbm_losses(rbm_losses, layers)
    plot_supervised_losses(supervised_losses)
    plot_metrics_comparison(results)
    plot_confusion_matrices([cms["Без"], cms["AE"], cms["RBM"]], ["Без", "AE", "RBM"], class_names or [str(i) for i in range(n_cls)])

    return results

results_cardio = run_experiment(
    dataset_name="Cardiotocography (NSP)",
    dataset_id=193,
    target_col='NSP',
    class_names=['N', 'S', 'P'],
    normalize_to_01=False
)

results_digits = run_experiment(
    dataset_name="Optical Recognition of Handwritten Digits",
    dataset_id=80,
    target_col=None,
    class_names=[str(i) for i in range(10)],
    normalize_to_01=True
)
