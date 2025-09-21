# src/predict_dl.py
import os
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # used if called from Streamlit; optional for CLI use
from src.utils import label_map

def load_model_tokenizer(model_dir="outputs/bert_sentiment", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_texts(texts, model_dir="outputs/bert_sentiment", batch_size=16):
    model, tokenizer, device = load_model_tokenizer(model_dir)
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
    return np.array(all_preds)

def evaluate_and_plot(y_true, y_pred, return_fig=False):
    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Build figure with side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_map[i] for i in range(3)],
                yticklabels=[label_map[i] for i in range(3)],
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')

    sns.countplot(x=y_pred, ax=axes[1])
    axes[1].set_xticks([0,1,2])
    axes[1].set_xticklabels([label_map[i] for i in range(3)])
    axes[1].set_title('Predicted Sentiment Distribution')

    # Optionally display metrics (when running in Streamlit)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if return_fig:
        return metrics, fig
    else:
        # If not returning fig, try to show with streamlit if available
        try:
            st.write("**Evaluation Metrics:**")
            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1-score: {f1:.3f}")
            st.pyplot(fig)
        except Exception:
            # fallback to returning metrics and fig
            return metrics, fig

def save_predictions_csv(texts, preds, out_path="outputs/predictions.csv"):
    import pandas as pd
    df = pd.DataFrame({"review": texts, "predicted_label": preds})
    df["predicted_label_name"] = df["predicted_label"].map(label_map)
    df.to_csv(out_path, index=False)
    return df
