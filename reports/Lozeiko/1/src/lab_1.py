import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Загрузка и подготовка данных
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Отделяем целевую переменную (класс) от признаков
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Заполнение пропущенных значений (если есть)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 2. PCA вручную с использованием numpy.linalg.eig
# Вычисляем ковариационную матрицу
cov_matrix = np.cov(X_scaled.T)

# Вычисляем собственные значения и собственные векторы
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Сортируем собственные значения и векторы по убыванию
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Выбираем первые две и три главные компоненты
PC2_manual = X_scaled.dot(eigenvectors_sorted[:, :2])
PC3_manual = X_scaled.dot(eigenvectors_sorted[:, :3])

# 3. PCA с использованием sklearn
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Получаем первые две и три главные компоненты
PC2_sklearn = X_pca[:, :2]
PC3_sklearn = X_pca[:, :3]

# 4. Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 2D визуализация (ручной метод)
scatter1 = axes[0, 0].scatter(PC2_manual[:, 0], PC2_manual[:, 1],
                             c=y, cmap='viridis', alpha=0.7)
axes[0, 0].set_title('2D PCA (Manual) - Первые две главные компоненты')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0, 0], label='Death Event')

# 2D визуализация (sklearn)
scatter2 = axes[0, 1].scatter(PC2_sklearn[:, 0], PC2_sklearn[:, 1],
                             c=y, cmap='viridis', alpha=0.7)
axes[0, 1].set_title('2D PCA (Sklearn) - Первые две главные компоненты')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[0, 1], label='Death Event')

# 3D визуализация (ручной метод)
ax3d1 = fig.add_subplot(2, 2, 3, projection='3d')
scatter3 = ax3d1.scatter(PC3_manual[:, 0], PC3_manual[:, 1], PC3_manual[:, 2],
                        c=y, cmap='viridis', alpha=0.7)
ax3d1.set_title('3D PCA (Manual) - Первые три главные компоненты')
ax3d1.set_xlabel('PC1')
ax3d1.set_ylabel('PC2')
ax3d1.set_zlabel('PC3')

# 3D визуализация (sklearn)
ax3d2 = fig.add_subplot(2, 2, 4, projection='3d')
scatter4 = ax3d2.scatter(PC3_sklearn[:, 0], PC3_sklearn[:, 1], PC3_sklearn[:, 2],
                        c=y, cmap='viridis', alpha=0.7)
ax3d2.set_title('3D PCA (Sklearn) - Первые три главные компоненты')
ax3d2.set_xlabel('PC1')
ax3d2.set_ylabel('PC2')
ax3d2.set_zlabel('PC3')

plt.tight_layout()
plt.show()

# 5. Расчет потерь информации
# Общая дисперсия
total_variance = np.sum(eigenvalues_sorted)

# Потери для 2 компонент
variance_2d = np.sum(eigenvalues_sorted[:2])
loss_2d = 1 - (variance_2d / total_variance)

# Потери для 3 компонент
variance_3d = np.sum(eigenvalues_sorted[:3])
loss_3d = 1 - (variance_3d / total_variance)

print("=== АНАЛИЗ ПОТЕРЬ ИНФОРМАЦИИ ===")
print(f"Общая дисперсия: {total_variance:.4f}")
print(f"Дисперсия первых 2 компонент: {variance_2d:.4f} ({variance_2d/total_variance*100:.1f}%)")
print(f"Потери информации (2D): {loss_2d:.4f} ({loss_2d*100:.1f}%)")
print(f"Дисперсия первых 3 компонент: {variance_3d:.4f} ({variance_3d/total_variance*100:.1f}%)")
print(f"Потери информации (3D): {loss_3d:.4f} ({loss_3d*100:.1f}%)")

# 6. Визуализация объясненной дисперсии
plt.figure(figsize=(10, 6))
explained_variance_ratio = eigenvalues_sorted / total_variance
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
        alpha=0.7, label='Индивидуальная объясненная дисперсия')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
         'ro-', label='Накопленная объясненная дисперсия')

plt.xlabel('Номер главной компоненты')
plt.ylabel('Объясненная дисперсия')
plt.title('Объясненная дисперсия главными компонентами')
plt.legend()
plt.grid(True)
plt.show()