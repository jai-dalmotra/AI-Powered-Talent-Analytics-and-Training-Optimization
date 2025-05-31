"""
sentiment_analysis.py
=====================
This module assigns sentiment scores to learner feedback using both TextBlob and VADER.
It outputs polarity scores and discrete sentiment labels.
"""

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------
# 🧪 Sentiment Scoring Functions
# --------------------------------

def get_textblob_polarity(text: str) -> float:
    """Returns polarity score between -1 and 1 using TextBlob."""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def get_vader_compound(text: str) -> float:
    """Returns compound sentiment score using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    try:
        return analyzer.polarity_scores(text)['compound']
    except Exception:
        return 0.0


# --------------------------------
# 🏷️ Sentiment Labeling
# --------------------------------

def label_sentiment(score: float, pos_thres=0.1, neg_thres=-0.1) -> str:
    """Convert sentiment score into label."""
    if score > pos_thres:
        return "Positive"
    elif score < neg_thres:
        return "Negative"
    else:
        return "Neutral"


# --------------------------------
# 🔄 Apply to DataFrame
# --------------------------------

def add_sentiment_columns(df: pd.DataFrame, text_column='cleaned_feedback') -> pd.DataFrame:
    """
    Adds sentiment scores and labels using both TextBlob and VADER.
    Returns enriched DataFrame.
    """
    df['tb_score'] = df[text_column].apply(get_textblob_polarity)
    df['tb_sentiment'] = df['tb_score'].apply(label_sentiment)

    df['vader_score'] = df[text_column].apply(get_vader_compound)
    df['vader_sentiment'] = df['vader_score'].apply(label_sentiment)

    print(f"✅ Added sentiment scores using TextBlob and VADER. Shape: {df.shape}")
    return df


# --------------------------------
# 📊 Aggregation for Recommender
# --------------------------------

def compute_avg_sentiment_per_trainer(df: pd.DataFrame, score_col='vader_score') -> pd.DataFrame:
    """
    Returns DataFrame with average sentiment per trainer.
    """
    return df.groupby('trainer_id')[score_col].mean().reset_index().rename(columns={score_col: 'avg_sentiment'})
