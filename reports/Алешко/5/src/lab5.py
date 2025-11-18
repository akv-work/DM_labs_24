import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

iris = datasets.load_iris()
X = iris.data
y = iris.target

print("Информация о датасете Iris")
print(f"Размерность признаков: {X.shape}")
print(f"Размерность целевой переменной: {y.shape}")
print(f"Названия классов: {iris.target_names}")
#Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

#Обучение моделей
#Одиночное дерево решений
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
#Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
#AdaBoost
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
#XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
#CatBoost
cat = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
cat.fit(X_train, y_train)

#Оценка точности
models = {
    'Decision Tree': dt,
    'Random Forest': rf,
    'AdaBoost': ada,
    'XGBoost': xgb,
    'CatBoost': cat
}

results = {}
print("\nТочность моделей на тестовой выборке")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

#Сравнение и вывод
print("\nСравнение моделей")
results_df = pd.DataFrame(list(results.items()), columns=['Модель', 'Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)
print(results_df)

best_model = results_df.iloc[0]['Модель']
best_acc = results_df.iloc[0]['Accuracy']

print(f"\nВывод: Лучшая модель — {best_model} с точностью {best_acc:.4f}.")
print("Все модели показали высокую точность, что ожидаемо для простого датасета Iris.")
print("Бустинговые методы (CatBoost, XGBoost, AdaBoost) и Random Forest обычно дают лучшие результаты, чем одиночное дерево.")