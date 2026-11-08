import streamlit as st
import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if required
nltk.download("stopwords", quiet=True)

# Load saved model
saved_data = joblib.load("models/best_news_model.pkl")

model = saved_data["model"]
tfidf = saved_data["tfidf"]
model_name = saved_data["model_name"]

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not", "no", "nor"}

stemmer = PorterStemmer()


# Text preprocessing
def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


# Streamlit page configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="centered"
)

# Application heading
st.title("📰 News Article Classification")

st.write(
    "Enter a news article below. "
    "Our NLP and Machine Learning model "
    "will automatically predict its category."
)

st.info(f"Model Used: {model_name} | Test Accuracy: 98.2%")

# User input
news_text = st.text_area(
    "Enter your news article:",
    height=200,
    placeholder="Paste your news article here..."
)

# Prediction button
if st.button("Predict Category", type="primary"):

    if not news_text.strip():

        st.warning("Please enter a news article first.")

    else:

        # Preprocess input
        cleaned_text = preprocess_text(news_text)

        if not cleaned_text.strip():
            st.warning("Please enter meaningful news text.")

        else:
            # Convert text into TF-IDF features
            text_vector = tfidf.transform([cleaned_text])

            # Predict category
            prediction = model.predict(text_vector)[0]

            st.success("Prediction Completed!")

            st.subheader("Predicted News Category")

            st.markdown(
                f"# 🏷️ {prediction.upper()}"
            )

            # Display cleaned text
            with st.expander("View Preprocessed Text"):
                st.write(cleaned_text)

# Footer
st.divider()

st.caption(
    "Developed using Python, NLP, TF-IDF and Machine Learning."
)