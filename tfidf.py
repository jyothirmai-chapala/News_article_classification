import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("dataset/bbc-text-cleaned.csv")

# Input and output
X = df["cleaned_text"]
y = df["category"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

# Fit on training data and transform
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

# Display results
print("TF-IDF FEATURE EXTRACTION")
print("-------------------------")

print("Training articles:", len(X_train))
print("Testing articles:", len(X_test))

print("\nTraining TF-IDF Shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF Shape:")
print(X_test_tfidf.shape)

print("\nTotal Features:")
print(len(tfidf.get_feature_names_out()))

print("\nSample Features:")
print(tfidf.get_feature_names_out()[:20])

print("\nSample TF-IDF Values:")
print(X_train_tfidf[0].toarray()[0][:20])

print("\nTF-IDF completed successfully!")