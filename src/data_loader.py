# src/data_loader.py
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_csv(path="data/reviews.csv"):
    df = pd.read_csv(path)
    # Ensure columns exist
    assert "review" in df.columns and "label" in df.columns, "CSV must contain 'review' and 'label' columns"
    return df

def make_hf_datasets(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))
    return train_ds, test_ds
