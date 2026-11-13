import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("dataset/bbc-text.csv")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))

# Keep negative words to preserve meaning
stop_words = stop_words - {"not", "no", "nor"}

stemmer = PorterStemmer()


# Text preprocessing function
def preprocess_text(text):

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove special characters, numbers and punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 3. Tokenization
    words = text.split()

    # 4. Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # 5. Stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    # Join words back into a sentence
    return " ".join(words)


# Apply preprocessing
df["cleaned_text"] = df["text"].apply(preprocess_text)


# Display original and cleaned text
print("\nORIGINAL TEXT:")
print(df["text"][0][:500])

print("\nCLEANED TEXT:")
print(df["cleaned_text"][0][:500])


# Save cleaned dataset
df.to_csv("dataset/bbc-text-cleaned.csv", index=False)

print("\nPreprocessing completed successfully!")
print("Cleaned dataset saved.")