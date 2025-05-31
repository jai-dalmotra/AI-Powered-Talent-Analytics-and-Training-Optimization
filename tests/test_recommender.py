# test_recommender.py
from src.recommender import prepare_surprise_data, train_svd_model, recommend_top_n
import pandas as pd


def test_svd_pipeline():
    df = pd.DataFrame({
        'learner_id': ['L1', 'L1', 'L2', 'L3'],
        'trainer_id': ['T1', 'T2', 'T2', 'T3'],
        'rating': [4, 5, 3, 4]
    })
    data = prepare_surprise_data(df)
    algo, rmse = train_svd_model(data)
    assert 0 < rmse <= 5

    all_trainers = ['T1', 'T2', 'T3', 'T4']
    rated = ['T1', 'T2']
    recs = recommend_top_n(algo, 'L1', all_trainers, rated, n=2)
    assert isinstance(recs, list)
    assert len(recs) == 2
