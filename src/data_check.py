import pandas as pd

# Load dataset
df = pd.read_csv("data/bank_churn.csv")

# Display first rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Column names
print("\nCOLUMNS:")
print(df.columns)

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Data types
print("\nDATA TYPES:")
print(df.dtypes)

# Churn distribution
print("\nTARGET DISTRIBUTION:")
print(df['Exited'].value_counts())