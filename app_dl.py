# app_dl.py
import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.predict_dl import load_model_tokenizer, predict_texts, evaluate_and_plot, save_predictions_csv
from src.utils import label_map

st.set_page_config(layout="wide", page_title="BERT Sentiment Analyzer (DL)")
st.title("Restaurant Review Sentiment Analyzer — Deep Learning (BERT fine-tune)")

# Load model/tokenizer
with st.spinner("Loading model..."):
    model, tokenizer, device = load_model_tokenizer("outputs/bert_sentiment")

# Sidebar single review predict
st.sidebar.header("Single Review Prediction")
user_review = st.sidebar.text_area("Enter review text")
if st.sidebar.button("Predict"):
    if user_review and user_review.strip():
        preds = predict_texts([user_review], model_dir="outputs/bert_sentiment")
        label = label_map[int(preds[0])]
        st.sidebar.success(f"Predicted: {label}")
    else:
        st.sidebar.warning("Please enter text")

# Main dataset section
st.header("Dataset & Batch Predictions")
df = load_csv("data/reviews.csv")
st.markdown(f"**Dataset size:** {len(df)}")
if st.button("Predict Dataset"):
    with st.spinner("Running predictions on dataset..."):
        preds = predict_texts(df['review'].tolist(), model_dir="outputs/bert_sentiment")
        df['predicted_label'] = preds
        df['predicted_label_name'] = df['predicted_label'].map(label_map)
        save_predictions_csv(df['review'].tolist(), preds, out_path="outputs/predictions.csv")
        st.success("Predictions saved to outputs/predictions.csv")
        st.dataframe(df.head(200))

        # Show metrics + plots side-by-side
        metrics, fig = evaluate_and_plot(df['label'].values, preds, return_fig=True)
        # Layout: left metrics, right plots
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Metrics")
            st.write(f"Accuracy: {metrics['accuracy']:.3f}")
            st.write(f"Precision: {metrics['precision']:.3f}")
            st.write(f"Recall: {metrics['recall']:.3f}")
            st.write(f"F1-score: {metrics['f1']:.3f}")
        with col2:
            st.pyplot(fig)
else:
    st.info("Click `Predict Dataset` to run batch predictions and view evaluation.")
