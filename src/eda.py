import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/bank_churn.csv")

# Style
sns.set(style="whitegrid")

print("DATASET SHAPE:")
print(df.shape)

print("\nCHURN DISTRIBUTION:")
print(df['Exited'].value_counts())

# =========================
# CHURN DISTRIBUTION
# =========================

plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)
plt.title("Customer Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.close()

# =========================
# GENDER VS CHURN
# =========================

plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Exited', data=df)
plt.title("Gender vs Churn")
plt.savefig("outputs/gender_vs_churn.png")
plt.close()

# =========================
# GEOGRAPHY VS CHURN
# =========================

plt.figure(figsize=(8,5))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title("Geography vs Churn")
plt.savefig("outputs/geography_vs_churn.png")
plt.close()

# =========================
# AGE DISTRIBUTION
# =========================

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("outputs/age_distribution.png")
plt.close()

# =========================
# BALANCE DISTRIBUTION
# =========================

plt.figure(figsize=(8,5))
sns.histplot(df['Balance'], bins=30, kde=True)
plt.title("Balance Distribution")
plt.savefig("outputs/balance_distribution.png")
plt.close()

# =========================
# CORRELATION HEATMAP
# =========================

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("\nEDA graphs saved successfully in outputs folder.")