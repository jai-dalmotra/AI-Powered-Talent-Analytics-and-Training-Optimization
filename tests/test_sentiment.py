import pandas as pd
from src.sentiment_analysis import add_sentiment_columns


def test_sentiment_labels():
    feedback_samples = ["This was fantastic!",
                        "Not great, could be better.",
                        "Just okay."]
    sample_df = pd.DataFrame({'cleaned_feedback': feedback_samples})
    result_df = add_sentiment_columns(sample_df)
    assert 'tb_sentiment' in result_df.columns
    assert 'vader_sentiment' in result_df.columns
    assert set(result_df['tb_sentiment'].unique()) <= {"Positive",
                                                       "Negative",
                                                       "Neutral"}
