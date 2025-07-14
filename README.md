# 🧠 NLP Disaster Tweet Classification with DistilBERT

This project focuses on predicting whether a tweet is related to a disaster using Natural Language Processing (NLP) techniques. A pretrained **DistilBERT** model is fine-tuned for binary sequence classification to label tweets as disaster-related or not. This project was submitted to a Kaggle competition.

---

## 🚀 Project Overview

- 🔍 **EDA & Preprocessing**: Cleaned tweets, removed URLs and special characters, and tokenized input using Hugging Face’s tokenizer.
- 🧪 **Model Training**: Fine-tuned a `distilbert-base-uncased` model on the training set using the Hugging Face Transformers library and PyTorch.
- 📊 **Evaluation**: Assessed performance on a validation set using **F1 score** (Kaggle’s official metric).
- 💾 **Model Saving**: Automatically saved the best model based on validation F1 score.
- 📉 **Early Stopping**: Prevented overfitting by monitoring validation performance and stopping when no further improvement was seen.

---

## 🛠️ Key Steps

### 1. Data Preprocessing
- Removed mentions, URLs, HTML tags, and redundant white spaces.
- Lowercased text and tokenized using `AutoTokenizer`.
- Converted encoded text and labels into PyTorch datasets and loaders.

### 2. Model Training and Evaluation
- Fine-tuned `distilbert-base-uncased` with:
  - `AdamW` optimizer
  - Linear learning rate scheduler
  - Batch size of 16
  - 2-epoch early stopping based on **validation F1**

---

## 📈 Training Results

| Epoch | Train Loss | Train Acc | Train F1 | Val Acc | Val F1 | Notes |
|-------|------------|-----------|----------|---------|--------|-------|
| 1     | 0.4430     | 0.8121    | 0.7581   | 0.8163  | 0.7929 | ✅ Best model saved |
| 2     | 0.3494     | 0.8587    | 0.8240   | 0.8176  | 0.7583 | ⚠️ No improvement |
| 3     | 0.2966     | 0.8851    | 0.8568   | 0.8255  | 0.7879 | ⚠️ No improvement — Early stop |

### 🔎 Observations

- The model reached **its best validation F1 score of 0.7929** during the **first epoch**.
- While training F1 and accuracy improved in later epochs, **validation performance plateaued**, suggesting **mild overfitting**.
- **Early stopping** prevented unnecessary training and saved the best-performing model.

---

## ✅ Key Features

- Transformer-based fine-tuning using **DistilBERT**
- **Evaluation-driven model checkpointing**
- **F1 score** optimization (aligned with Kaggle scoring)
- Training tracked with **progress bars (tqdm)** for clean logging

---

## 🧠 Next Steps

- 🔧 **Tune hyperparameters**: learning rate, batch size, weight decay
- 🧼 **Use additional features**: like `location`
- 🧪 **Try other models**: like RoBERTa, or even logistic regression for comparison
- 📦 **Deploy with Streamlit**: build an interactive demo using the trained model

---

## Important Notice

The code in this repository is proprietary and protected by copyright law. Unauthorized copying, distribution, or use of this code is strictly prohibited. By accessing this repository, you agree to the following terms:

- **Do Not Copy:** You are not permitted to copy any part of this code for any purpose.
- **Do Not Distribute:** You are not permitted to distribute this code, in whole or in part, to any third party.
- **Do Not Use:** You are not permitted to use this code, in whole or in part, for any purpose without explicit permission from the owner.

If you have any questions or require permission, please contact the repository owner.

Thank you for your cooperation.

