import pandas as pd
import os
import re
import string
from typing import Optional


def load_csv(filepath: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {filepath} - shape: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None


def profile_df(df: pd.DataFrame, name: str = "Data") -> None:
    print(f"\n📌 Profile: {name}")
    print(df.info())
    print("\n🧪 Null Values:")
    print(df.isnull().sum())
    print("\n📊 Sample Rows:")
    print(df.head())


def validate_columns(df: pd.DataFrame, required_cols: list, name: str = "Data") -> bool:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in {name}: {missing}")
        return False
    return True


def clean_text(text: str) -> str:
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_feedback_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'CustomerID': 'learner_id',
        'Country': 'trainer_id',  # Mapping 'Country' to trainer ID
        'FeedbackScore': 'rating'
    })

    # 🛠️ Convert textual ratings to numeric
    rating_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    df["rating"] = df["rating"].map(rating_map)

    # Drop rows where rating conversion failed (e.g., unexpected text)
    df = df.dropna(subset=["rating"])

    if 'feedback_text' not in df.columns:
        df['feedback_text'] = "Feedback score: " + df['rating'].astype(str)

    required_cols = ['learner_id', 'trainer_id', 'rating', 'feedback_text']
    if not validate_columns(df, required_cols, name="Feedback Data"):
        return df

    df = df.dropna(subset=['learner_id', 'trainer_id', 'rating'])
    df['feedback_text'] = df['feedback_text'].fillna("")
    df['cleaned_feedback'] = df['feedback_text'].apply(clean_text)
    df = df.drop_duplicates(subset=['learner_id', 'trainer_id', 'feedback_text'])

    print(f"✅ Feedback data cleaned. Shape: {df.shape}")
    return df


def preprocess_trainer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'DBN': 'trainer_id',
        'School Name': 'name'
    })

    df['domain'] = "School"  # Placeholder domain

    required_cols = ['trainer_id', 'name', 'domain']
    if not validate_columns(df, required_cols, name="Trainer Data"):
        return df

    df = df.dropna(subset=['trainer_id', 'name'])
    df['domain'] = df['domain'].fillna('Unknown')
    df = df.drop_duplicates(subset=['trainer_id'])

    print(f"✅ Trainer data cleaned. Shape: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_csv(path, index=False)
        print(f"💾 Saved cleaned data to {path}")
    except Exception as e:
        print(f"❌ Failed to save {path}: {e}")
