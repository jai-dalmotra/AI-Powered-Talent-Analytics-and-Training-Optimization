from src.data_loader import load_csv, preprocess_feedback_df, preprocess_trainer_df, save_processed
from src.sentiment_analysis import add_sentiment_columns, compute_avg_sentiment_per_trainer
import os


os.makedirs("data", exist_ok=True)


feedback_raw = load_csv("data/session_feedback.csv")
trainer_raw = load_csv("data/trainer_profiles.csv")


feedback_clean = preprocess_feedback_df(feedback_raw)
trainer_clean = preprocess_trainer_df(trainer_raw)


feedback_sentiment = add_sentiment_columns(feedback_clean)


avg_sentiment_df = compute_avg_sentiment_per_trainer(feedback_sentiment)


save_processed(feedback_sentiment, "data/preprocessed_feedback.csv")
save_processed(trainer_clean, "data/preprocessed_trainers.csv")
save_processed(avg_sentiment_df, "data/avg_sentiment_per_trainer.csv")

print("✅ All processed data files saved in /data")
