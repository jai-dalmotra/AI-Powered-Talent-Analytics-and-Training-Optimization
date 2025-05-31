# tests/test_data_loader.py
from src.data_loader import load_csv, preprocess_feedback_df


def test_feedback_loading():
    df = load_csv("data/session_feedback.csv")
    assert df is not None, "Failed to load session_feedback.csv"
    assert 'feedback_text' in df.columns


def test_feedback_preprocessing():
    df = load_csv("data/session_feedback.csv")
    df_clean = preprocess_feedback_df(df)
    assert 'cleaned_feedback' in df_clean.columns
    assert df_clean['cleaned_feedback'].apply(lambda x: isinstance(x, str)).all()
