# 🧠 Disaster Tweet Classification – ML & NLP Comparison + Streamlit Deployment

This project classifies tweets as disaster-related or not using both **traditional ML models** and a **fine-tuned DistilBERT transformer model**. It was originally developed for the [Kaggle NLP Disaster Tweets Competition](https://www.kaggle.com/competitions/nlp-getting-started) and includes a deployed **Streamlit app** for real-time prediction.

---

## 🚀 Project Highlights

- ✅ Built and evaluated **3 models**: Naive Bayes, Logistic Regression, and DistilBERT
- 🧪 Preprocessed tweet data using regex cleaning and keyword merging
- 📊 Compared performance across models (F1 score was the main metric)
- 🖥️ Deployed the final classifier using **Streamlit** for interactive use
- 🥇 **DistilBERT performed best** on the validation set and Kaggle leaderboard

---

## 📦 Models Compared

| Model              | Validation F1 | Kaggle Score | Notes                          |
|-------------------|---------------|--------------|--------------------------------|
| Naive Bayes        | ~0.78         | ~0.79        | Simple, fast baseline model using TF-IDF; performs reasonably well.           |
| Logistic Regression| ~0.78         | ~0.79        | Slightly more stable than NB; comparable performance with better calibration.        |
| DistilBERT         | **0.79+**     | **0.83**| Fine-tuned transformer; outperformed both baselines on validation and leaderboard.   |

---

## 🧹 Preprocessing

- Removed:
  - User mentions (`@user`)
  - URLs and HTML tags
  - Special characters and extra whitespace
- Lowercased text
- Merged `keyword` and `text` fields during training

---

## 🧠 DistilBERT Fine-Tuning

- Base model: `distilbert-base-uncased`
- Trained using:
  - `AdamW` optimizer
  - Learning rate scheduler
  - Early stopping (based on validation F1)
- Used Hugging Face `transformers` + PyTorch
- Best validation F1: **0.7929**

---

## 🌐 Live Demo – Streamlit App

Try out the classifier in your browser:  
👉 **[Disaster Tweet Classifier (Streamlit App)](https://disastertweetclassifier.streamlit.app/)**

> Choose a model (Naive Bayes or Logistic Regression), enter a tweet, and classify it in real time.

### 🔧 Models in the App:
- ✅ Naive Bayes
- ✅ Logistic Regression
- ✅ DistilBERT

---

## 🧪 Next Steps

- 📈 Display model metrics (F1, ROC AUC) in app sidebar
- 🧹 Improve text preprocessing pipeline using `nltk` or `spacy`
- 🔄 Add additional features like `location` or tweet metadata
- 🚀 Optimize DistilBERT loading time with quantization or other model compression techniques

---

## 📚 Tools & Libraries

- **Streamlit** for UI deployment
- **scikit-learn** for traditional ML models
- **Hugging Face Transformers** for DistilBERT
- **PyTorch** for fine-tuning
- **Pandas**, **Joblib**, **TQDM**, **Regex** for preprocessing and utilities

---

## Important Notice

The code in this repository is proprietary and protected by copyright law. Unauthorized copying, distribution, or use of this code is strictly prohibited. By accessing this repository, you agree to the following terms:

- **Do Not Copy:** You are not permitted to copy any part of this code for any purpose.
- **Do Not Distribute:** You are not permitted to distribute this code, in whole or in part, to any third party.
- **Do Not Use:** You are not permitted to use this code, in whole or in part, for any purpose without explicit permission from the owner.

If you have any questions or require permission, please contact the repository owner.

Thank you for your cooperation.

