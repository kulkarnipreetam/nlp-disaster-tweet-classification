import streamlit as st
import joblib
import os
import re
import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# App title
st.set_page_config(page_title="Disaster Tweet Classifier")
st.title("🚨 Disaster Tweet Classifier")
st.write("Classify tweets as real disaster-related or not using different models.")

# Model selection
model_choice = st.selectbox("Choose a model:", ["Naive Bayes", "Logistic Regression", "DistilBERT"])

# Tweet input
tweet = st.text_area("Enter tweet text:")

# === Preprocessing function ===
def clean_text(text):
    text = re.sub(r"@\w+", "", text)         # Remove @mentions
    text = re.sub(r"http\S+", "", text)      # Remove URLs
    text = re.sub(r"<.*?>", "", text)        # Remove HTML tags
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text)         # Remove extra whitespace
    return text.strip().lower()

# === Load Naive Bayes ===
@st.cache_resource
def load_naive_bayes_model():
    model = joblib.load(os.path.join("naive_bayes_model", "model.pkl"))
    vectorizer = joblib.load(os.path.join("naive_bayes_model", "vectorizer.pkl"))
    return model, vectorizer

# === Load Logistic Regression ===
@st.cache_resource
def load_logreg_model():
    model = joblib.load(os.path.join("logistic_regression_model", "model.pkl"))
    vectorizer = joblib.load(os.path.join("logistic_regression_model", "vectorizer.pkl"))
    return model, vectorizer

# === Load distilbert ===
@st.cache_resource
def load_distilbert_model():
    model_path = "preetamkulkarni/distilbert_disaster_tweet"  # Update this path as needed
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    return model, tokenizer

# === Predict ===
if st.button("Classify"):
    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:
        # Preprocess tweet
        cleaned_tweet = clean_text(tweet)

        # Load model/vectorizer
        if model_choice == "Naive Bayes":
            model, vectorizer = load_naive_bayes_model()
            X = vectorizer.transform([cleaned_tweet])
            pred = model.predict(X)[0]
            confidence = max(model.predict_proba(X)[0]) if hasattr(model, "predict_proba") else 1.0

        elif model_choice == "Logistic Regression":
            model, vectorizer = load_logreg_model()
            X = vectorizer.transform([cleaned_tweet])
            pred = model.predict(X)[0]
            confidence = max(model.predict_proba(X)[0]) if hasattr(model, "predict_proba") else 1.0

        else:  # DistilBERT
            model, tokenizer = load_distilbert_model()
            inputs = tokenizer(cleaned_tweet, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                confidence = probabilities[pred]

        label = "🔥 Real Disaster" if pred == 1 else "💬 Not a Disaster"
        st.success(f"**Prediction:** {label}")
        st.info(f"**Model:** {model_choice} | **Confidence:** {confidence:.2f}")
