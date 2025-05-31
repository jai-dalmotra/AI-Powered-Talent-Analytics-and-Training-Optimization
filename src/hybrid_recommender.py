

def hybrid_recommend_top_n(algo, learner_id: str, all_trainers: list, rated_trainers: list,
                           trainer_sentiment_df, weight_rating=0.7, weight_sentiment=0.3, n=5):
    """
    Returns Top-N trainers for a learner using:
    - Predicted SVD rating
    - Average sentiment score per trainer
    - Weighted hybrid scoring
    """

    # Filter unseen trainers
    unseen = [t for t in all_trainers if t not in rated_trainers]

    # Create a dict for quick lookup of sentiment
    sentiment_dict = dict(zip(trainer_sentiment_df['trainer_id'], trainer_sentiment_df['avg_sentiment']))

    results = []
    for trainer_id in unseen:
        rating = algo.predict(learner_id, trainer_id).est
        sentiment = sentiment_dict.get(trainer_id, 0.0)
        hybrid_score = (weight_rating * rating) + (weight_sentiment * sentiment * 5)  # normalize sentiment to 5
        results.append((trainer_id, hybrid_score))

    top_n = sorted(results, key=lambda x: x[1], reverse=True)[:n]
    return top_n
