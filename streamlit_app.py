import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import joblib
import os
import re
import torch
import warnings

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Suppress known FutureWarnings from huggingface and transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

for key in ["tweet_input", "model_choice", "prediction", "confidence", "label"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.markdown("### 📈 Visualization Settings", unsafe_allow_html=True)
    all_metrics = ["ROC Curve", "Precision-Recall", "Reliability Curve", "F1 vs. Threshold"]

    selected_metric = st.radio("Choose metric to display:", all_metrics)

# App title
st.set_page_config(page_title="Disaster Tweet Classifier", layout="wide")
st.title("🚨 Disaster Tweet Classifier")
st.write("Classify tweets as real disaster-related or not using different models.")

col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    # Model selection
    model_choice = st.selectbox("Choose a model:", ["Naive Bayes", "Logistic Regression", "DistilBERT"],
                                index=["Naive Bayes", "Logistic Regression", "DistilBERT"].index(st.session_state.model_choice) if st.session_state.model_choice else 0)


    # Tweet input
    tweet = st.text_area("Enter tweet text:", value=st.session_state.tweet_input if st.session_state.tweet_input else "")

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
        model = AutoModelForSequenceClassification.from_pretrained(model_path, token=hf_token)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
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
            
            # Store results in session state
            st.session_state.tweet_input = tweet
            st.session_state.model_choice = model_choice
            st.session_state.prediction = pred
            st.session_state.confidence = confidence
            st.session_state.label = label
            
            if st.session_state.prediction is not None:
                st.success(f"**Prediction:** {st.session_state.label}")
                st.info(f"**Model:** {st.session_state.model_choice} | **Confidence:** {st.session_state.confidence:.2f}")

with col2:      
    @st.cache_data
    def load_metrics():
        models = ["nb", "lr", "distilbert"]
        data = {}
        for model in models:
            df = pd.read_csv(f"utils/{model}_metrics.csv")
            data[model] = df
        return data

    metrics_data = load_metrics()

    model_names = {"nb": "Naive Bayes", "lr": "Logistic Regression", "distilbert": "DistilBERT"}
    colors = {"nb": "#1f77b4", "lr": "#2ca02c", "distilbert": "#d62728"} 

    def plot_roc_plotly(metrics_data, model_names, colors):
        fig = go.Figure()
        for model_key, df in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=df["fpr"],
                y=df["tpr"],
                mode="lines",
                name=f"{model_names[model_key]} (AUC = {df['roc_auc'][0]:.2f})",
                line=dict(color=colors[model_key]),
                hovertemplate = ('<span style="color:{color};">{name}</span><br>' 
                                'FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>').format(color=colors[model_key], name=model_names[model_key])
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False
        ))
        fig.update_layout(
            title=dict(text="ROC Curve", x=0.5, xanchor='center'),
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            ),
            margin=dict(t=50, b=80),
            template="plotly_white"
        )
        return fig

    def plot_pr_plotly(metrics_data, model_names, colors):
        fig = go.Figure()
        for model_key, df in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=df["recall"],
                y=df["precision"],
                mode="lines",
                name=model_names[model_key],
                line=dict(color=colors[model_key]),
                hovertemplate = ('<span style="color:{color};">{name}</span><br>' 
                                'Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>').format(color=colors[model_key], name=model_names[model_key])
            ))
        fig.update_layout(
            title=dict(text="Precision-Recall Curve", x=0.5, xanchor='center'),
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            ),
            margin=dict(t=50, b=80),
            template="plotly_white"
        )
        return fig

    def plot_reliability_plotly(metrics_data, model_names, colors):
        fig = go.Figure()
        for model_key, df in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=df["bin_center"],
                y=df["true_rate"],
                mode="lines+markers",
                name=f"{model_names[model_key]} (Brier = {df['brier_score'][0]:.3f})",
                line=dict(color=colors[model_key]),
                marker=dict(symbol="circle", size=8),
                hovertemplate = ('<span style="color:{color};">{name}</span><br>' 
                                'Predicted Prob: %{{x:.3f}}<br>True Rate: %{{y:.3f}}<extra></extra>').format(color=colors[model_key], name=model_names[model_key])
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False
        ))
        fig.update_layout(
            title=dict(text="Reliability Curve", x=0.5, xanchor='center'),
            xaxis_title="Predicted Probability",
            yaxis_title="Empirical True Rate",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            ),
            margin=dict(t=50, b=80),
            template="plotly_white"
        )
        return fig

    def plot_f1_plotly(metrics_data, model_names, colors):
        fig = go.Figure()
        for model_key, df in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=df["pr_thresholds"],
                y=df["f1_score"],
                mode="lines",
                name= model_names[model_key],
                line=dict(color=colors[model_key]),
                hovertemplate = ('<span style="color:{color};">{name}</span><br>' 
                                'Threshold: %{{x:.2f}}<br>F1 Score: %{{y:.2f}}<extra></extra>').format(color=colors[model_key], name=model_names[model_key])
            ))

        fig.update_layout(
            title=dict(text="F1 score vs. Threshold", x=0.5, xanchor='center'),
            xaxis_title="Threshold",
            yaxis_title="F1 Score",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            ),
            margin=dict(t=50, b=80),
            template="plotly_white"
        )
        return fig

    if selected_metric == "ROC Curve":
        roc_fig = plot_roc_plotly(metrics_data, model_names, colors)
        st.plotly_chart(roc_fig, use_container_width=True, height = 500)

    if selected_metric == "Precision-Recall":
        pr_fig = plot_pr_plotly(metrics_data, model_names, colors)
        st.plotly_chart(pr_fig, use_container_width=True, height = 500)

    if selected_metric == "Reliability Curve":
        rel_fig = plot_reliability_plotly(metrics_data, model_names, colors)
        st.plotly_chart(rel_fig, use_container_width=True, height = 500)
        
    if selected_metric == "F1 vs. Threshold":
        rel_fig = plot_f1_plotly(metrics_data, model_names, colors)
        st.plotly_chart(rel_fig, use_container_width=True, height = 500)
