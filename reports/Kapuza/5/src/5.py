import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score

# Импорт библиотек бустинга
# Примечание: Убедитесь, что они установлены (pip install xgboost catboost)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Задание 1: Загрузить встроенный набор данных digits 
digits = load_digits()
X = digits.data
y = digits.target

print(f"Размерность данных: {X.shape}")
print(f"Количество классов: {len(set(y))} (цифры от 0 до 9)")
print("-" * 30)

#  Задание 2: Разделить данные на обучающую и тестовую выборки 
# Обычно используют разбиение 80/20 или 70/30. Возьмем 80/20.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Задание 3: Инициализация моделей 
models = {
    "Decision Tree (Одиночное дерево)": DecisionTreeClassifier(random_state=42),
    "Random Forest (Случайный лес)": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42), # algorithm='SAMME' лучше работает для мультикласса в старых версиях, в новых auto
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42), # verbose=0 отключает вывод логов обучения
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
}

results = {}

print("Начало обучения и оценки моделей...\n")

#  Задания 3 и 4: Обучение и вывод classification_report 
for name, model in models.items():
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = model.predict(X_test)
    
    # Расчет точности (accuracy) для итогового сравнения
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    # Вывод отчета (Задание 4)
    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 60)

#  Задание 5: Сравнение общей точности и вывод 
print("Итоговое сравнение точности (Accuracy):")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for name, acc in sorted_results:
    print(f"{name}: {acc:.4f}")

best_model = sorted_results[0][0]
print(f"\n>> ВЫВОД: Лучше всего для этой задачи подходит модель: {best_model}")