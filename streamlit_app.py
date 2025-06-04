# app/streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import preprocess_feedback_df, load_csv
from src.sentiment_analysis import add_sentiment_columns
from src.recommender import prepare_surprise_data, train_svd_model
from src.hybrid_recommender import hybrid_recommend_top_n
from src.visualization import (
    plot_sentiment_distribution,
    plot_avg_rating_per_trainer,
    plot_sentiment_vs_rating,
    plot_learner_engagement,
    plot_learner_journey
)

# 🎛️ Page Config
st.set_page_config(
    page_title="Talent Analytics Dashboard",
    page_icon="🎓",
    layout="wide"
)

# 🖼️ Inject Custom CSS and Logo
st.markdown("""
<style>
body {
    background-color: #0a0a0a;
    color: #f0e68c;
    font-family: 'Segoe UI', sans-serif;
}

section.main > div:has(~ footer ) {
    padding-top: 1rem;
    background: linear-gradient(to bottom right, #000000, #111111);
    border-radius: 25px;
    box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
    margin-top: 2rem;
    padding: 2rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #FFD700;
    font-weight: 700;
}

.stButton > button {
    background-color: gold;
    color: black;
    font-weight: bold;
    border-radius: 12px;
    padding: 8px 20px;
    box-shadow: 0 0 5px rgba(255, 215, 0, 0.6);
}

.css-1v0mbdj p {
    font-size: 14px;
    color: #ccc;
}

.stDataFrame thead tr th {
    background-color: #333;
    color: #FFD700;
}

div[data-testid="stMetricValue"] {
    color: gold;
    font-weight: bold;
    font-size: 1.8em;
}

.sidebar .sidebar-content {
    background-color: #1c1c1c;
    border-radius: 15px;
    padding: 15px;
    color: #FFD700;
}

input, .stSelectbox, .stSlider {
    background-color: #1a1a1a;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.image("assets/luxe_banner.png", use_column_width=True)
st.title("🎓 AI-Powered Feedback & Recommendation System")

st.title("🎓 AI-Powered Feedback & Recommendation System")

# ─────────────────────────────────────────────────────────────
# 📥 LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading and processing feedback data..."):
    feedback_df = load_csv("data/session_feedback.csv")

    if feedback_df.empty:
        st.error("❌ `session_feedback.csv` is missing or empty. Please check your data folder.")
        st.stop()

    feedback_df = preprocess_feedback_df(feedback_df)
    feedback_df = add_sentiment_columns(feedback_df)

# ─────────────────────────────────────────────────────────────
# 📊 DASHBOARD METRICS
# ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Key Metrics Overview")
col1, col2, col3 = st.columns(3)
col1.metric("📋 Feedback Records", len(feedback_df))
col2.metric("🧑‍🎓 Unique Learners", feedback_df['learner_id'].nunique())
col3.metric("🧑‍🏫 Unique Trainers", feedback_df['trainer_id'].nunique())

# ─────────────────────────────────────────────────────────────
# 🤖 TRAINING RECOMMENDER MODEL
# ─────────────────────────────────────────────────────────────
with st.spinner("🤖 Training SVD-based recommender model..."):

    # 🔍 Debugging: Check rating column
    st.write("🔍 Sample ratings (first 10):", feedback_df["rating"].unique()[:10])

    # Proceed with model training
    data = prepare_surprise_data(feedback_df)
    algo, _ = train_svd_model(data)

# Compute average sentiment per trainer
trainer_sentiment_df = (
    feedback_df.groupby("trainer_id")["vader_score"]
    .rename(columns={"vader_score": "avg_sentiment"})
)

# ─────────────────────────────────────────────────────────────
# 🔧 SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Recommender Controls")

learner_ids = sorted(feedback_df["learner_id"].unique().tolist())
selected_learner = st.sidebar.selectbox("👤 Choose a Learner", learner_ids)

weight_rating = st.sidebar.slider("⚖️ Weight for Ratings", 0.0, 1.0, 0.6, step=0.05)
weight_sentiment = 1.0 - weight_rating
n_recommendations = st.sidebar.slider("📌 Number of Recommendations", 1, 10, 5)

# ─────────────────────────────────────────────────────────────
# 🎯 GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
rated_trainers = feedback_df[feedback_df["learner_id"] == selected_learner]["trainer_id"].tolist()
all_trainers = feedback_df["trainer_id"].unique().tolist()

recommendations = hybrid_recommend_top_n(
    algo=algo,
    learner_id=selected_learner,
    all_trainers=all_trainers,
    rated_trainers=rated_trainers,
    trainer_sentiment_df=trainer_sentiment_df,
    weight_rating=weight_rating,
    weight_sentiment=weight_sentiment,
    n=n_recommendations
)

# ─────────────────────────────────────────────────────────────
# 📄 DISPLAY RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
st.markdown(f"## 🎯 Top {n_recommendations} Recommendations for Learner `{selected_learner}`")

rec_df = pd.DataFrame(recommendations, columns=["Trainer ID", "Hybrid Score"])
st.dataframe(rec_df.style.format({"Hybrid Score": "{:.2f}"}), use_container_width=True)

st.download_button(
    label="📥 Download as CSV",
    data=rec_df.to_csv(index=False).encode("utf-8"),
    file_name=f"recommendations_{selected_learner}.csv",
    mime="text/csv"
)


# 🔧 Dark mode settings for plots
plt.style.use("dark_background")
sns.set_style("darkgrid", {
    'axes.facecolor': '#222222',
    'figure.facecolor': '#222222'
})

# ─────────────────────────────────────────────────────────────
# 📊 DASHBOARD TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Sentiment Distribution",
    "⭐ Avg Ratings by Trainer",
    "📈 Rating vs Sentiment",
    "👥 Learner Engagement",
    "🧭 Learner Journey",
    "🧑‍🏫 Trainer Profiles",
    "🏫 School Summaries",
    "🗃️ Raw Data Preview"
])

with tabs[0]:
    st.subheader("📊 Overall Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0a")
    sns.countplot(x="tb_sentiment", data=feedback_df, palette="pastel", ax=ax)

    ax.set_title("TextBlob Sentiment Count", color='#FFD700', fontsize=14)
    ax.set_xlabel("Sentiment", color='#FFD700')
    ax.set_ylabel("Count", color='#FFD700')
    ax.tick_params(colors='#FFD700')

    fig.patch.set_facecolor("#0a0a0a")
    fig.tight_layout()
    st.pyplot(fig)

with tabs[1]:
    st.subheader("🏆 Top Trainers by Average Rating")

    top_trainers = (
        feedback_df.groupby("trainer_id")["rating"]
        .mean().sort_values(ascending=False).head(10)
    )

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0a")
    sns.barplot(x=top_trainers.values, y=top_trainers.index, palette="viridis", ax=ax)

    ax.set_title("Top 10 Trainers by Average Rating", color='#FFD700', fontsize=14)
    ax.set_xlabel("Average Rating", color='#FFD700')
    ax.set_ylabel("Trainer ID", color='#FFD700')
    ax.tick_params(colors='#FFD700')

    fig.patch.set_facecolor("#0a0a0a")
    fig.tight_layout()
    st.pyplot(fig)

with tabs[2]:
    st.subheader("📈 Sentiment vs Rating")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0a")
    sns.boxplot(x="tb_sentiment", y="rating", data=feedback_df, palette="coolwarm", ax=ax)

    ax.set_title("Rating Distribution by Sentiment", color='#FFD700', fontsize=14)
    ax.set_xlabel("Sentiment", color='#FFD700')
    ax.set_ylabel("Rating", color='#FFD700')
    ax.tick_params(colors='#FFD700')

    fig.patch.set_facecolor("#0a0a0a")
    fig.tight_layout()
    st.pyplot(fig)

with tabs[3]:
    st.subheader("👥 Learner Engagement Levels")

    learner_counts = feedback_df["learner_id"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0a")
    sns.histplot(learner_counts, bins=20, kde=False, color="#FFD700", ax=ax)

    ax.set_title("Distribution of Feedbacks per Learner", color='#FFD700', fontsize=14)
    ax.set_xlabel("Number of Feedbacks", color='#FFD700')
    ax.set_ylabel("Number of Learners", color='#FFD700')
    ax.tick_params(colors='#FFD700')

    fig.patch.set_facecolor("#0a0a0a")
    fig.tight_layout()
    st.pyplot(fig)

with tabs[4]:
    st.subheader(f"🧭 Journey for Learner `{selected_learner}`")

    learner_df = feedback_df[feedback_df["learner_id"] == selected_learner]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a0a0a")
    sns.lineplot(data=learner_df, x=learner_df.index, y="rating", marker="o", color="#FFD700", ax=ax)

    ax.set_title("Feedback Ratings Over Time", color='#FFD700', fontsize=14)
    ax.set_xlabel("Session Index", color='#FFD700')
    ax.set_ylabel("Rating", color='#FFD700')
    ax.tick_params(colors='#FFD700')

    fig.patch.set_facecolor("#0a0a0a")
    fig.tight_layout()
    st.pyplot(fig)

with tabs[5]:
    st.subheader("🗃️ Raw Feedback Data")
    st.dataframe(feedback_df.head(50), use_container_width=True)

with tabs[7]:
    st.subheader("📂 Complete Feedback Snapshot")

    st.markdown("""
    <style>
    .dataframe tbody tr:hover {
        background-color: #333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(
        feedback_df.style
            .highlight_max(axis=0, color='#FFD700')
            .set_properties(**{
                'background-color': '#111111',
                'color': '#FFD700',
                'border-color': 'gray'
            }),
        use_container_width=True,
        height=500
    )

# ─────────────────────────────────────────────────────────────
# 👣 FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built by Jai Dalmotra | Talent Analytics 2025")
