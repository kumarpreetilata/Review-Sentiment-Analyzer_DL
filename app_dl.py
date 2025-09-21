import streamlit as st
import pandas as pd
from src.data_loader import load_csv
from src.predict_dl import load_model_tokenizer, predict_texts, evaluate_and_plot, save_predictions_csv
from src.utils import label_map

st.set_page_config(layout="wide", page_title="BERT Sentiment Analyzer (DL)")
st.title("Restaurant Review Sentiment Analyzer — BERT Fine-tune")

# Load model/tokenizer
with st.spinner("Loading model..."):
    model, tokenizer, device = load_model_tokenizer("outputs/bert_sentiment")

# -----------------------
# Sidebar: Single Review
# -----------------------
st.sidebar.header("Single Review Prediction")
user_review = st.sidebar.text_area("Enter review text")
if st.sidebar.button("Predict"):
    if user_review and user_review.strip():
        preds, confs = predict_texts([user_review], model_dir="outputs/bert_sentiment", return_confidence=True)
        label = label_map[int(preds[0])]
        st.sidebar.success(f"Predicted: {label} (Confidence: {confs[0]*100:.1f}%)")
    else:
        st.sidebar.warning("Please enter review text")

# -----------------------
# Main: Dataset & Batch Predictions
# -----------------------
st.header("Dataset & Batch Predictions")
df = load_csv("data/reviews.csv")
st.markdown(f"**Dataset size:** {len(df)}")

# Restaurant filter
restaurant_filter = st.selectbox("Filter by Restaurant (optional)", ["All"] + df['restaurant'].unique().tolist())
if restaurant_filter != "All":
    filtered_df = df[df['restaurant'] == restaurant_filter]
else:
    filtered_df = df

if st.button("Predict Dataset"):
    with st.spinner("Running predictions on dataset..."):
        texts = filtered_df['review'].tolist()
        preds, confs = predict_texts(texts, model_dir="outputs/bert_sentiment", return_confidence=True)
        filtered_df['predicted_label'] = preds
        filtered_df['predicted_label_name'] = [label_map[p] for p in preds]
        filtered_df['confidence'] = confs
        save_predictions_csv(texts, preds, confidences=confs, out_path="outputs/predictions.csv")
        st.success("Predictions saved to outputs/predictions.csv")
        st.dataframe(filtered_df)

        # Show metrics + plots
        if 'label' in filtered_df.columns:
            metrics, fig = evaluate_and_plot(filtered_df['label'].values, preds, return_fig=True)
            col1, col2 = st.columns([1,2])
            with col1:
                st.subheader("Metrics")
                st.write(f"Accuracy: {metrics['accuracy']:.3f}")
                st.write(f"Precision: {metrics['precision']:.3f}")
                st.write(f"Recall: {metrics['recall']:.3f}")
                st.write(f"F1-score: {metrics['f1']:.3f}")
            with col2:
                st.pyplot(fig)
        else:
            st.info("No ground truth labels available for metrics.")

else:
    st.info("Click `Predict Dataset` to run batch predictions and view evaluation.")
