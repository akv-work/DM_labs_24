import pandas as pd
import numpy as np
import os
import kagglehub
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

print("Downloading dataset via KaggleHub...")
dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
file_path = os.path.join(dataset_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Dataset downloaded to: {file_path}")

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути {file_path}")
    exit()

df = df.drop('customerID', axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

df_processed = df.copy()
df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

model_results = {}

tree_model = DecisionTreeClassifier(random_state=42, max_depth=5)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
f1_tree = f1_score(y_test, y_pred_tree, pos_label=1)
model_results['Decision Tree'] = f1_tree

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, pos_label=1)
model_results['Random Forest'] = f1_rf

ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
f1_ada = f1_score(y_test, y_pred_ada, pos_label=1)
model_results['AdaBoost'] = f1_ada

xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
f1_xgb = f1_score(y_test, y_pred_xgb, pos_label=1)
model_results['XGBoost'] = f1_xgb

X_cat = df.drop('Churn', axis=1)
y_cat = df['Churn']
categorical_features = X_cat.select_dtypes(include=['object']).columns.tolist()

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_cat, y_cat, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_cat
)

cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    random_seed=42,
    cat_features=categorical_features,
    verbose=0
)
cat_model.fit(X_train_cat, y_train_cat)
y_pred_cat = cat_model.predict(X_test_cat)
f1_cat = f1_score(y_test_cat, y_pred_cat, pos_label=1)
model_results['CatBoost'] = f1_cat

print("\n--- Model F1-Scores (Class: Churn=1) ---")
results_df = pd.DataFrame(
    model_results.items(), 
    columns=['Model', 'F1-Score']
).sort_values(by='F1-Score', ascending=False)

print(results_df)

print("\n--- Detailed Report for Best Model (CatBoost) ---")
print(classification_report(y_test_cat, y_pred_cat, target_names=['Not Churn (0)', 'Churn (1)']))