import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/bank_churn.csv")

# =========================
# DROP USELESS COLUMNS
# =========================

df.drop(['CustomerId', 'Surname', 'Year'], axis=1, inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================

df['BalanceSalaryRatio'] = (
    df['Balance'] / (df['EstimatedSalary'] + 1)
)

df['ProductPerTenure'] = (
    df['NumOfProducts'] / (df['Tenure'] + 1)
)

df['EngagementScore'] = (
    df['IsActiveMember'] *
    df['NumOfProducts']
)

df['AgeTenureInteraction'] = (
    df['Age'] * df['Tenure']
)

# =========================
# ENCODING
# =========================

df = pd.get_dummies(
    df,
    columns=['Geography', 'Gender'],
    drop_first=True
)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop('Exited', axis=1)
y = df['Exited']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# =========================
# TRAIN & EVALUATE
# =========================

for name, model in models.items():

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Probabilities
    y_prob = model.predict_proba(X_test_scaled)[:,1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Print Metrics
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))