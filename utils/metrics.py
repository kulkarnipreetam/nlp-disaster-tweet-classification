import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, brier_score_loss
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import joblib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Load Data 
df = pd.read_csv("./train.csv")
X = df['text']
y = df['target']

# Train-Test Split with Stratify 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_configs = [
    {
        "name": "nb",
        "load_model": lambda: joblib.load("../naive_bayes_model/model.pkl"),
        "load_vectorizer": lambda: joblib.load("../naive_bayes_model/vectorizer.pkl"),
        "predict_proba": lambda model, X_input: model.predict_proba(X_input)[:, 1],
        "transform": lambda vec, X: vec.transform(X)
    },
    {
        "name": "lr",
        "load_model": lambda: joblib.load("../logistic_regression_model/model.pkl"),
        "load_vectorizer": lambda: joblib.load("../logistic_regression_model/vectorizer.pkl"),
        "predict_proba": lambda model, X_input: model.predict_proba(X_input)[:, 1],
        "transform": lambda vec, X: vec.transform(X)
    },
    {
        "name": "distilbert",
        "load_model": lambda: AutoModelForSequenceClassification.from_pretrained("preetamkulkarni/distilbert_disaster_tweet"),
        "load_vectorizer": lambda: AutoTokenizer.from_pretrained("preetamkulkarni/distilbert_disaster_tweet"),
        "predict_proba": lambda model, tokenizer: get_distilbert_probs(model, tokenizer, X_test),
        "transform": lambda vec, X: vec
    }
]

def get_distilbert_probs(model, tokenizer, texts):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            all_probs.append(probs[0][1].item())
    return np.array(all_probs)


def pad(arr, length):
    arr = np.array(arr, dtype=float)
    return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)

for config in model_configs:
    print(f"Processing model: {config['name']}")

    model = config["load_model"]()
    vectorizer = config["load_vectorizer"]()

    X_input = config["transform"](vectorizer, X_test)
    probs = config["predict_proba"](model, X_input)

    # Curve metrics
    fpr, tpr, roc_thresholds = roc_curve(y_test, probs)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, probs)
    f1score = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # Reliability bins
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(probs, bins) - 1
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    true_rate = [np.mean(y_test[bin_indices == i]) if np.any(bin_indices == i) else np.nan for i in range(10)]
    counts = [np.sum(bin_indices == i) for i in range(10)]

    # Summary metrics
    auc_score = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    max_len = max(len(fpr), len(precisions), len(bin_centers))

    # Build DataFrame
    df_metrics = pd.DataFrame({
        "fpr": pad(fpr, max_len),
        "tpr": pad(tpr, max_len),
        "roc_thresholds": pad(roc_thresholds, max_len),
        "precision": pad(precisions, max_len),
        "recall": pad(recalls, max_len),
        "f1_score": pad(f1score, max_len),
        "pr_thresholds": pad(pr_thresholds, max_len),
        "bin_center": pad(bin_centers, max_len),
        "true_rate": pad(true_rate, max_len),
        "count": pad(counts, max_len),
        "roc_auc": pad([auc_score] , max_len),
        "brier_score": pad([brier] , max_len)
    })

    # Save
    df_metrics.to_csv(f"{config['name']}_metrics.csv", index=False)
    print(f"Saved {config['name']}_metrics.csv")

