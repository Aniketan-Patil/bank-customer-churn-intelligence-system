import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/bank_churn.csv")

# =========================
# DROP NON-INFORMATIVE COLUMNS
# =========================

df.drop(['CustomerId', 'Surname', 'Year'], axis=1, inplace=True)

# =========================
# FEATURE ENGINEERING
# =========================

# Balance to Salary Ratio
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

# Products per Tenure
df['ProductPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)

# Engagement Score
df['EngagementScore'] = (
    df['IsActiveMember'] *
    df['NumOfProducts']
)

# Age-Tenure Interaction
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
# SPLIT FEATURES & TARGET
# =========================

X = df.drop('Exited', axis=1)
y = df['Exited']

# =========================
# TRAIN-TEST SPLIT
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
# OUTPUT INFORMATION
# =========================

print("Preprocessing Completed Successfully")

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nFinal Features:")
print(X.columns.tolist())