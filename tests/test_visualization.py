import pandas as pd
from src.visualization import (
    plot_sentiment_distribution,
    plot_avg_rating_per_trainer,
    plot_sentiment_vs_rating,
    plot_learner_journey
)


def test_visuals_run_without_errors():
    df = pd.DataFrame({
        'learner_id': ['L1']*3 + ['L2'],
        'trainer_id': ['T1', 'T2', 'T3', 'T1'],
        'rating': [4, 5, 3, 4],
        'vader_sentiment': ['Positive', 'Neutral', 'Negative', 'Positive'],
        'vader_score': [0.8, 0.0, -0.5, 0.7],
        'tb_sentiment': ['Positive', 'Neutral', 'Negative', 'Positive']
    })

    plot_sentiment_distribution(df)
    plot_avg_rating_per_trainer(df)
    plot_sentiment_vs_rating(df)
    learner_id = plot_learner_journey(df)
    assert learner_id in df['learner_id'].values
