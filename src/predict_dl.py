import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st  # optional fallback
from src.utils import label_map

def load_model_tokenizer(model_dir="outputs/bert_sentiment", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_texts(texts, model_dir="outputs/bert_sentiment", batch_size=16, return_confidence=False):
    model, tokenizer, device = load_model_tokenizer(model_dir)
    all_preds = []
    all_conf = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=64, return_tensors="pt")
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds.tolist())
            if return_confidence:
                conf = np.max(probs, axis=-1)
                all_conf.extend(conf.tolist())

    if return_confidence:
        return np.array(all_preds), np.array(all_conf)
    return np.array(all_preds)

def evaluate_and_plot(y_true, y_pred, return_fig=False):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

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

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if return_fig:
        return metrics, fig
    else:
        try:
            st.write("**Evaluation Metrics:**")
            st.write(f"Accuracy: {acc:.3f}")
            st.write(f"Precision: {prec:.3f}")
            st.write(f"Recall: {rec:.3f}")
            st.write(f"F1-score: {f1:.3f}")
            st.pyplot(fig)
        except Exception:
            return metrics, fig

def save_predictions_csv(texts, preds, confidences=None, out_path="outputs/predictions.csv"):
    import pandas as pd
    df = pd.DataFrame({"review": texts, "predicted_label": preds})
    df["predicted_label_name"] = df["predicted_label"].map(label_map)
    if confidences is not None:
        df["confidence"] = confidences
    df.to_csv(out_path, index=False)
    return df
