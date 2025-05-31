"""
recommender.py
==============
This module builds and applies a collaborative filtering-based recommendation system using Surprise.
"""

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


# --------------------------
# 📦 Data Wrapping (Surprise)
# --------------------------

def prepare_surprise_data(df: pd.DataFrame, user_col='learner_id', item_col='trainer_id', rating_col='rating'):
    """
    Converts DataFrame into Surprise dataset format.
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[user_col, item_col, rating_col]], reader)
    return data


# --------------------------
# 🔁 Train SVD Recommender
# --------------------------

def train_svd_model(data):
    """
    Trains an SVD collaborative filtering model.
    Returns trained model and test RMSE.
    """
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    return algo, rmse


# --------------------------
# 🎯 Recommend Top N Trainers
# --------------------------

def recommend_top_n(algo, learner_id: str, all_trainers: list, rated_trainers: list, n: int = 5):
    """
    Returns Top-N trainer recommendations for a learner based on predicted rating.
    """
    # Predict only for trainers not already rated
    unseen = [t for t in all_trainers if t not in rated_trainers]
    predictions = [(t, algo.predict(learner_id, t).est) for t in unseen]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return top_n
