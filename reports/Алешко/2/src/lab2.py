import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

data = load_breast_cancer()
X = data.data  
y = data.target  
feature_names = data.feature_names
target_names = data.target_names

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Автоэнкодер
class Autoencoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(model, data_loader, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_loader):.4f}")

dataset = TensorDataset(X_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
ae_2d = Autoencoder(input_dim=30, latent_dim=2)
train_autoencoder(ae_2d, data_loader)
X_ae_2d = ae_2d.encode(X_tensor).detach().numpy()
ae_3d = Autoencoder(input_dim=30, latent_dim=3)
train_autoencoder(ae_3d, data_loader)
X_ae_3d = ae_3d.encode(X_tensor).detach().numpy()

def plot_2d(X_proj, y, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], format=plt.FuncFormatter(lambda val, loc: target_names[int(val)]))
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def plot_3d(X_proj, y, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=y, cmap='viridis', alpha=0.7)
    fig.colorbar(scatter, ticks=[0, 1], format=plt.FuncFormatter(lambda val, loc: target_names[int(val)]))
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.show()

plot_2d(X_ae_2d, y, 'Autoencoder 2D Projection')
plot_3d(X_ae_3d, y, 'Autoencoder 3D Projection')

# t-SNE
perplexities = [20, 30, 40, 50, 60]
best_perplexity = 50  

tsne_2d = TSNE(n_components=2, perplexity=best_perplexity, init='pca', random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)
plot_2d(X_tsne_2d, y, f't-SNE 2D (perplexity={best_perplexity})')
tsne_3d = TSNE(n_components=3, perplexity=best_perplexity, init='pca', random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)
plot_3d(X_tsne_3d, y, f't-SNE 3D (perplexity={best_perplexity})')

for perp in perplexities:
    tsne_temp = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42)
    X_temp = tsne_temp.fit_transform(X_scaled)
    plot_2d(X_temp, y, f't-SNE 2D (perplexity={perp})') 

# PCA
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print(f"Explained variance (2D PCA): {pca_2d.explained_variance_ratio_}")
plot_2d(X_pca_2d, y, 'PCA 2D Projection')
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
print(f"Explained variance (3D PCA): {pca_3d.explained_variance_ratio_}")
plot_3d(X_pca_3d, y, 'PCA 3D Projection')
