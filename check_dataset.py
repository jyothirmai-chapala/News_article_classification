import pandas as pd

# Load dataset
df = pd.read_csv("dataset/bbc-text.csv")

# Display first 5 records
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display category counts
print("\nCategory Distribution:")
print(df["category"].value_counts())