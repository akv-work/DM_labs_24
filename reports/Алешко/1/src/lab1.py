import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_PATH = "D:\универ\универ\ИАД\seeds_dataset.txt"  
OUTPUT_DIR = "pca_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_seeds(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python', header=None)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
    return df

df = load_seeds(DATA_PATH)
print("Исходный размер данных:", df.shape)

n_cols = df.shape[1]
data = df.iloc[:, :n_cols-1].copy()  # признаки
labels = df.iloc[:, n_cols-1].copy()  # класс

# Обработка пропусков 
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(data)
X = pd.DataFrame(X_imputed, columns=data.columns)

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)  # numpy array

X_mean = scaler.mean_
X_std = scaler.scale_

# Вариант 1: вручную 
cov_matrix = np.cov(X_scaled, rowvar=False)  
eigvals, eigvecs = np.linalg.eig(cov_matrix)
sort_idx = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[sort_idx]
eigvecs_sorted = eigvecs[:, sort_idx]
W2 = eigvecs_sorted[:, :2] 
W3 = eigvecs_sorted[:, :3] 
X_proj_manual_2 = X_scaled.dot(W2)  
X_proj_manual_3 = X_scaled.dot(W3)  

# Вариант 2: sklearn PCA 
pca2 = PCA(n_components=2)
X_proj_skl_2 = pca2.fit_transform(X_scaled)
pca3 = PCA(n_components=3)
X_proj_skl_3 = pca3.fit_transform(X_scaled)

# Сравнение собственного решения и sklearn 
unique_labels = np.unique(labels)
markers = ['o', 's', '^', 'v', 'P', '*', 'X', 'D']  

def plot_2d(X2, labels, title, filename):
    plt.figure(figsize=(7,6))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        plt.scatter(X2[mask,0], X2[mask,1],
                    label=str(lab),
                    marker=markers[i % len(markers)],
                    s=40)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title="class")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=200)
    print("Сохранено:", out)
    plt.close()

def plot_3d(X3, labels, title, filename):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(X3[mask,0], X3[mask,1], X3[mask,2],
                   label=str(lab),
                   marker=markers[i % len(markers)],
                   s=40)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(title="class", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print("Сохранено:", out)
    plt.close()

# manual
plot_2d(X_proj_manual_2, labels.values, "Manual PCA: projection on first 2 PCs", "manual_pca_2d.png")
plot_3d(X_proj_manual_3, labels.values, "Manual PCA: projection on first 3 PCs", "manual_pca_3d.png")
# sklearn
plot_2d(X_proj_skl_2, labels.values, "sklearn PCA: projection on first 2 PCs", "sklearn_pca_2d.png")
plot_3d(X_proj_skl_3, labels.values, "sklearn PCA: projection on first 3 PCs", "sklearn_pca_3d.png")

# Вычисление потерь
total_variance = np.sum(eigvals_sorted.real) 
explained_by_2 = np.sum(eigvals_sorted.real[:2])
explained_by_3 = np.sum(eigvals_sorted.real[:3])

retained_pct_2 = explained_by_2 / total_variance * 100.0
retained_pct_3 = explained_by_3 / total_variance * 100.0
loss_pct_2 = 100.0 - retained_pct_2
loss_pct_3 = 100.0 - retained_pct_3

print(f"Суммарная дисперсия (всех компонент): {total_variance:.6f}")
print(f"Суммарная доля, удержанная 2-мя PC: {retained_pct_2:.4f}%  => потери {loss_pct_2:.4f}%")
print(f"Суммарная доля, удержанная 3-мя PC: {retained_pct_3:.4f}%  => потери {loss_pct_3:.4f}%")

# Реконструкция и MSE 
X_recon_2 = X_proj_manual_2.dot(W2.T)  
X_recon_3 = X_proj_manual_3.dot(W3.T)

X_recon_2_orig = scaler.inverse_transform(X_recon_2)
X_recon_3_orig = scaler.inverse_transform(X_recon_3)
X_orig = scaler.inverse_transform(X_scaled)

mse_2 = np.mean((X_orig - X_recon_2_orig)**2)
mse_3 = np.mean((X_orig - X_recon_3_orig)**2)
var_total = np.mean((X_orig - X_orig.mean(axis=0))**2)  

print(f"MSE reconstruction (2-PC): {mse_2:.6f}")
print(f"MSE reconstruction (3-PC): {mse_3:.6f}")
print(f"Отн. потеря дисперсии (2-PC): {mse_2/var_total*100:.4f}%")
print(f"Отн. потеря дисперсии (3-PC): {mse_3/var_total*100:.4f}%")


