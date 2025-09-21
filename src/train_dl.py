import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
import evaluate

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("data/reviews.csv").dropna(subset=["review", "label"])

# -----------------------
# Stratified Train/Test Split
# -----------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]  # preserves label distribution
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# -----------------------
# Tokenizer
# -----------------------
MODEL_NAME = "distilbert-base-uncased"  # fast & lightweight
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["review"], padding=True, truncation=True)  # dynamic padding

tokenized_train = train_ds.map(tokenize, batched=True)
tokenized_test = test_ds.map(tokenize, batched=True)

# -----------------------
# Model
# -----------------------
label2id = {0:0, 1:1, 2:2}
id2label = {0:"Negative",1:"Neutral",2:"Positive"}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# -----------------------
# Metrics
# -----------------------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# -----------------------
# Training Arguments
# -----------------------
training_args = TrainingArguments(
    output_dir="outputs/bert_sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,          # longer training for small dataset
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
)

# -----------------------
# Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # stops if f1 doesn't improve
)

# -----------------------
# Train
# -----------------------
trainer.train()

# -----------------------
# Save Model & Tokenizer
# -----------------------
os.makedirs("outputs/bert_sentiment", exist_ok=True)
trainer.save_model("outputs/bert_sentiment")
tokenizer.save_pretrained("outputs/bert_sentiment")

print("Training complete. Model saved to outputs/bert_sentiment/")
