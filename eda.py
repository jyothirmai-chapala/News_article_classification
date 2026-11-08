import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/bbc-text.csv")

# 1. Dataset information
print("DATASET INFORMATION")
print("-------------------")
print("Total articles:", len(df))
print("Total categories:", df["category"].nunique())

# 2. Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# 3. Check duplicate records
print("\nDUPLICATE RECORDS")
print("Duplicates:", df.duplicated().sum())

# 4. Count words in each article
df["word_count"] = df["text"].apply(
    lambda text: len(str(text).split())
)

print("\nWORD COUNT STATISTICS")
print(df["word_count"].describe())

# 5. Display one sample article
print("\nSAMPLE NEWS ARTICLE")
print("Category:", df.iloc[0]["category"])
print("Text:", df.iloc[0]["text"][:500])

# 6. Category distribution chart
category_counts = df["category"].value_counts()

plt.figure(figsize=(9, 5))
category_counts.plot(kind="bar")

plt.title("News Article Category Distribution")
plt.xlabel("News Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 7. Article word count distribution
plt.figure(figsize=(9, 5))
plt.hist(df["word_count"], bins=30)

plt.title("Distribution of Article Word Counts")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.show()