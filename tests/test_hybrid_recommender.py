# tests/test_hybrid_recommender.py

from src.recommender import prepare_surprise_data, train_svd_model
from src.hybrid_recommender import hybrid_recommend_top_n
import pandas as pd


def test_hybrid_recommendation_output():
    # Prepare mock feedback data
    feedback_data = pd.DataFrame({
        'learner_id': ['L1', 'L2', 'L3', 'L1'],
        'trainer_id': ['T1', 'T1', 'T2', 'T3'],
        'rating': [5, 3, 4, 2]
    })

    # Prepare average sentiment scores for trainers
    sentiment_df = pd.DataFrame({
        'trainer_id': ['T1', 'T2', 'T3'],
        'avg_sentiment': [0.8, 0.4, 0.9]  # normalized polarity (VADER)
    })

    # Train collaborative filtering model
    data = prepare_surprise_data(feedback_data)
    model, _ = train_svd_model(data)

    # Get hybrid recommendations
    learner = 'L1'
    all_trainers = ['T1', 'T2', 'T3']
    rated_trainers = ['T1', 'T3']  # already rated by L1

    results = hybrid_recommend_top_n(
        model, learner, all_trainers, rated_trainers,
        trainer_sentiment_df=sentiment_df,
        weight_rating=0.6,
        weight_sentiment=0.4,
        n=2
    )

    # Validate results
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
    assert all(isinstance(item[1], float) for item in results)
