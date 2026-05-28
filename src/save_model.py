import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/bank_churn.csv")

# =========================
# DROP NON-INFORMATIVE COLUMNS
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
# ENCODE CATEGORICAL VARIABLES
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

# =========================
# MODEL TRAINING
# =========================

model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

# =========================
# SAVE MODEL & SCALER
# =========================

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Model and scaler saved successfully.")