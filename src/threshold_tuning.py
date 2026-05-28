import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
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
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# SCALE
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

# =========================
# PROBABILITIES
# =========================

y_prob = model.predict_proba(X_test_scaled)[:,1]

# =========================
# CUSTOM THRESHOLD
# =========================

threshold = 0.35

y_pred = (y_prob >= threshold).astype(int)

# =========================
# METRICS
# =========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nThreshold Used: {threshold}")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")