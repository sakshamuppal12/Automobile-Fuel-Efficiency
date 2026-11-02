import pandas as pd

# Load dataset
df = pd.read_csv("data/epa2021.csv")

print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nTarget Variable Distribution:")
print(df["comb_mpg"].describe())
