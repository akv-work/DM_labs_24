import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

try:
    data = pd.read_csv("diabetes.csv")
except FileNotFoundError:
    print("Файл не найден")

X = data.drop(["Outcome"], axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print("-" * 30)

scores = {}

tree = DecisionTreeClassifier(random_state=42, max_depth=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
rec_tree = recall_score(y_test, y_pred_tree)
scores["Decision Tree"] = rec_tree
print(f"Decision Tree Recall: {rec_tree:.4f}")

rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rec_rf = recall_score(y_test, y_pred_rf)
scores["Random Forest"] = rec_rf
print(f"Random Forest Recall: {rec_rf:.4f}")

ada = AdaBoostClassifier(n_estimators=200, random_state=42, algorithm='SAMME')
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
rec_ada = recall_score(y_test, y_pred_ada)
scores["AdaBoost"] = rec_ada
print(f"AdaBoost Recall:      {rec_ada:.4f}")

cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    silent=True,
    random_seed=42
)
cat.fit(X_train, y_train)
y_pred_cat = cat.predict(X_test)
rec_cat = recall_score(y_test, y_pred_cat)
scores["CatBoost"] = rec_cat
print(f"CatBoost Recall:      {rec_cat:.4f}")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rec_xgb = recall_score(y_test, y_pred_xgb)
scores["XGBoost"] = rec_xgb
print(f"XGBoost Recall:       {rec_xgb:.4f}")

print("-" * 30)
results = pd.DataFrame(list(scores.items()), columns=["Model", "Recall"])
print("\nИтоговая таблица (сортировка по Recall):")
print(results.sort_values("Recall", ascending=False))

best_model = results.sort_values("Recall", ascending=False).iloc[0]
print(f"\nЛучшая модель по метрике Recall: {best_model['Model']} ({best_model['Recall']:.4f})")