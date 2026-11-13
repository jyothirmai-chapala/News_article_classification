import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load cleaned dataset
df = pd.read_csv("dataset/bbc-text-cleaned.csv")

# Input and output
X = df["cleaned_text"]
y = df["category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF feature extraction
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}
best_accuracy = 0
best_model = None
best_model_name = ""

# Train and evaluate models
for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Find best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Display comparison
print("\nMODEL COMPARISON")
print("----------------")

for name, accuracy in results.items():
    print(name, ":", round(accuracy * 100, 2), "%")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")

# Save best model and TF-IDF vectorizer together
joblib.dump(
    {
        "model": best_model,
        "tfidf": tfidf,
        "model_name": best_model_name
    },
    "models/best_news_model.pkl"
)

print("\nBest model saved successfully!")

# Plot model comparison
plt.figure(figsize=(9, 5))

plt.bar(results.keys(), results.values())

plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("ML Algorithms")
plt.ylabel("Accuracy")

plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Plot confusion matrix for best model
best_predictions = best_model.predict(X_test_tfidf)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=best_model.classes_
)

plt.figure(figsize=(8, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.xticks(range(len(best_model.classes_)), best_model.classes_, rotation=45)
plt.yticks(range(len(best_model.classes_)), best_model.classes_)

plt.colorbar()
plt.tight_layout()
plt.show()