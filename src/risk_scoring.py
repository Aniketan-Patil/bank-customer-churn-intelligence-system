import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# RANDOM FOREST MODEL
# =========================

model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

# =========================
# CHURN PROBABILITIES
# =========================

churn_probabilities = model.predict_proba(X_test_scaled)[:,1]

# =========================
# CUSTOM THRESHOLD
# =========================

threshold = 0.35

# =========================
# RISK SEGMENTATION
# =========================

risk_levels = []

for prob in churn_probabilities:

    if prob < 0.25:
        risk_levels.append("Low Risk")

    elif prob < 0.50:
        risk_levels.append("Medium Risk")

    elif prob < 0.75:
        risk_levels.append("High Risk")

    else:
        risk_levels.append("Critical Risk")

# =========================
# PREDICTED CHURN FLAG
# =========================

predicted_churn = (
    churn_probabilities >= threshold
).astype(int)

# =========================
# RESULTS DATAFRAME
# =========================

results = pd.DataFrame({
    'ChurnProbability': churn_probabilities,
    'PredictedChurn': predicted_churn,
    'RiskLevel': risk_levels
})

# =========================
# DISPLAY RESULTS
# =========================

print("\nFIRST 10 RISK SCORES:")
print(results.head(10))

# =========================
# SAVE RESULTS
# =========================

results.to_csv(
    "outputs/churn_risk_scoring.csv",
    index=False
)

# =========================
# FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES:")
print(feature_importance.head(10))

# =========================
# FEATURE IMPORTANCE PLOT
# =========================

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title("Top 10 Feature Importance")

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()

# =========================
# RISK DISTRIBUTION PLOT
# =========================

plt.figure(figsize=(8,5))

sns.countplot(
    x='RiskLevel',
    data=results,
    order=[
        'Low Risk',
        'Medium Risk',
        'High Risk',
        'Critical Risk'
    ]
)

plt.title("Customer Risk Distribution")

plt.savefig(
    "outputs/risk_distribution.png"
)

plt.close()

print("\nRisk scoring completed successfully.")