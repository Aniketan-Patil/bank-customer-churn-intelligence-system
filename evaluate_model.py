import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("data/bank_churn.csv")

# Feature engineering (same as project)

df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["ProductPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
df["EngagementScore"] = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

# Encoding
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

X = df.drop(
    columns=[
        "Exited",
        "CustomerId",
        "Surname",
        "Year"
    ]
)

y = df["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))