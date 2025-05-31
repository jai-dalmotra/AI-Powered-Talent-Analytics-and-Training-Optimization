# visualization.py
# =================
# This module contains reusable functions for plotting KPIs and trends
# from the feedback and recommendation system.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", palette="muted")


def plot_sentiment_distribution(df: pd.DataFrame):
    """
    Dual sentiment distribution: TextBlob and VADER.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x='tb_sentiment', data=df, ax=axs[0], hue='tb_sentiment', palette='Set2', legend=False)
    axs[0].set_title("TextBlob Sentiment Distribution")
    sns.countplot(x='vader_sentiment', data=df, ax=axs[1], hue='vader_sentiment', palette='coolwarm', legend=False)
    axs[1].set_title("VADER Sentiment Distribution")
    plt.suptitle("Sentiment Analysis Overview", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_avg_rating_per_trainer(df: pd.DataFrame):
    """
    Bar plot of average rating per trainer (top 10).
    """
    top_trainers = df.groupby('trainer_id')['rating'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=top_trainers.index, y=top_trainers.values, hue=top_trainers.index, palette='crest', legend=False)
    plt.title("Top 10 Trainers by Average Rating")
    plt.ylabel("Avg Rating")
    plt.xlabel("Trainer ID")
    plt.xticks(rotation=45)
    plt.ylim(3, 5)
    plt.tight_layout()
    plt.show()


def plot_sentiment_vs_rating(df: pd.DataFrame):
    """
    Scatter plot: Sentiment score (VADER) vs Rating.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='vader_score', y='rating', hue='vader_sentiment', alpha=0.7)
    plt.title("VADER Sentiment vs Rating", fontsize=13)
    plt.xlabel("Sentiment Score (VADER)")
    plt.ylabel("Feedback Rating")
    plt.tight_layout()
    plt.show()


def plot_learner_journey(df: pd.DataFrame):
    """
    Visualizes feedback ratings given by the most active learner.
    """
    learner_id = df['learner_id'].value_counts().idxmax()
    learner_data = df[df['learner_id'] == learner_id]

    plt.figure(figsize=(8, 4))
    sns.barplot(data=learner_data, x='trainer_id', y='rating', hue='vader_sentiment', dodge=False)
    plt.title(f"Ratings by Learner: {learner_id}")
    plt.ylabel("Rating")
    plt.xlabel("Trainer")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.show()
    return learner_id


def plot_learner_engagement(df: pd.DataFrame):
    """
    Histogram: Sessions attended per learner.
    """
    learner_sessions = df.groupby('learner_id').size()
    plt.figure(figsize=(8, 4))
    sns.histplot(learner_sessions, bins=15, kde=False, color='teal')
    plt.title("Learner Engagement (Sessions Attended)")
    plt.xlabel("Number of Sessions")
    plt.ylabel("Learner Count")
    plt.tight_layout()
    plt.show()
