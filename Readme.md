# Restaurant Review Sentiment Analyzer (Multi-class)

Analyze customer reviews for restaurants to detect **Positive**, **Neutral**, and **Negative** sentiment using **TF-IDF + Logistic Regression** locally, or a more advanced **DistilBERT-based classifier** for higher accuracy. Fully interactive via **Streamlit**.

---

## 🔧 Tools & Frameworks

<details>
<summary>Click to expand</summary>

- **Python** – main programming language  
- **Pandas & NumPy** – data manipulation  
- **Scikit-learn** – TF-IDF vectorizer, Logistic Regression, train/test split  
- **PyTorch & Transformers (HuggingFace)** – DistilBERT model for sequence classification  
- **Datasets & Evaluate (HuggingFace)** – Dataset handling, metrics (accuracy, precision, recall, F1)  
- **Streamlit** – interactive web application  
- **OS / Logging** – file handling and experiment tracking

</details>

---

## 🗂 Project Structure

restaurant_sentiment_local/
│
├── data/
│ └── reviews.csv # Sample dataset with review & label
│
├── src/
│ ├── data_loader.py # Load & preprocess data
│ ├── sentiment_model.py # Train & save TF-IDF + Logistic Regression
│ ├── predict.py # Predict sentiment and evaluate model
│ └── utils.py # Helper functions
│
├── outputs/
│ ├── sentiment_model.pkl # Saved TF-IDF + LR model & vectorizer
│ └── predictions.csv # Optional saved predictions
│
├── app.py # Streamlit web app for interactive use
├── main.py # Train & test pipeline (CLI)
└── requirements.txt



---

## ⚡ Features

- Multi-class sentiment classification (**Positive / Neutral / Negative**)  
- **Local TF-IDF + Logistic Regression** model for lightweight inference  
- Optional **DistilBERT classifier** for more accurate results using HuggingFace  
- **Interactive Streamlit app** for live review testing  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score  
- Fully reproducible and modular for other datasets

---

## 📊 HuggingFace DistilBERT Implementation Highlights

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import pandas as pd

# Load dataset
df = pd.read_csv("data/reviews.csv").dropna(subset=["review","label"])

# Tokenizer & model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=3)

# Metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")
```
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Train TF-IDF + Logistic Regression model
python -m src.train_dl

# 4. Run Streamlit app
streamlit run app_dl.py

Training uses Trainer with early stopping and metrics tracking

Saves model & tokenizer in outputs/bert_sentiment
