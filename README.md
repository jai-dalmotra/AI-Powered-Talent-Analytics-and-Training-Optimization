# Talent-Analytics-and-Training-Optimization
##  🧠 AI-Powered Feedback & Recommendation System for Gig-Based EdTech Platforms

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app-name>.streamlit.app)

Welcome to the **full-stack ML/NLP project** designed to automate and personalize training experiences in EdTech using learner feedback. This repository contains all code, configurations, notebooks, and documentation to:

- Analyze sentiment in learner feedback 📝
- Recommend the best trainer for each learner 👩‍🏫
- Visualize KPIs and feedback using an interactive dashboard 📊

## 🚀 Project Overview

> This project simulates a real-world system where learners provide textual feedback and star ratings after training sessions. The system analyzes that feedback to identify satisfaction levels and then recommends trainers using collaborative filtering and sentiment-aware ranking.

---

## 🗂️ Repository Structure

```bash
📁 edtech-recommender/
├── 📁 data/                        # Input and processed datasets
│   ├── session_feedback.csv       # Feedback data: learner_id, trainer_id, rating, feedback_text
│   ├── trainer_profiles.csv       # Trainer data: trainer_id, name, domain, experience
│   └── preprocessed_feedback.csv  # Cleaned + sentiment-labeled feedback
│
├── 📁 src/                         # Core Python modules
│   ├── data_loader.py             # Functions to load, clean, and preprocess CSVs
│   ├── sentiment_analysis.py      # TextBlob/VADER-based sentiment scoring
│   ├── recommender.py             # Collaborative Filtering with Surprise (SVD/KNN)
│   ├── hybrid_recommender.py      # Recommender enhanced with sentiment weight
│   ├── visualization.py           # Charts and KPIs (matplotlib, seaborn)
│   └── dashboard_helpers.py       # Streamlit helper functions (selects, filters, etc.)
│
├── 📁 notebooks/                  # Jupyter notebooks for exploration and validation
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_modeling.ipynb
│   ├── 03_recommender_training.ipynb
│   ├── 04_hybrid_modeling.ipynb
│   └── 05_dashboard_preview.ipynb
│
├── 📁 tests/                  # Jupyter notebooks for tests
│   ├── test_sentiment.py
│   ├── test_recommender.py
│   ├── test_hybrid_recommender.py
│   └── tests_data_loader.py
│
├── 📁 app/                         # Streamlit frontend app
│   └── streamlit_app.py           # Complete dashboard with interactivity
│
├── 📁 reports/                     # Project documentation and report assets
│   ├── final_report.pdf           # Academic-style technical report
│   └── screenshots/               # Dashboard and chart screenshots
│
├── 📄 README.md                    # You're here!
├── 📄 requirements.txt             # All required Python libraries
└── 📄 .gitignore                   # Ignore checkpoints, data, pycache, etc.
```

---

## 🔧 Features (Modules)

| Module                  | Description |
|------------------------|-------------|
| **Feedback Cleaning**  | Normalizes text (lowercase, punctuation removal, stopword cleaning) |
| **Sentiment Analysis** | Assigns polarity using TextBlob and VADER classifiers |
| **Recommendation**     | Uses collaborative filtering (SVD) from Surprise to match trainers |
| **Hybrid Engine**      | Reranks recommendations using average sentiment weight per trainer |
| **Visualization**      | Plots: sentiment histograms, trainer KPIs, learner rating trends |
| **Dashboard**          | Built with Streamlit, supports user input, visual charts, and results |

---

## ⚙️ Installation

Clone the repo:
```bash
git clone https://github.com/jai-dalmotra/Talent-Analytics-and-Training-Optimization.git
cd Talent-Analytics-and-Training-Optimization
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Datasets:
- Kaggle feedback dataset: https://www.kaggle.com/datasets/aditirai2607/feedback-dataset
- Trainer dataset: https://www.kaggle.com/datasets/jahnavipaliwal/customer-feedback-and-satisfaction

Run Jupyter for exploration:
```bash
jupyter lab
```

Or launch the Streamlit dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 How It Works — Step-by-Step

1. **Load Data**
   - Raw feedback and trainer profiles are loaded using `data_loader.py`
   - Preprocessing is applied to clean feedback and format text fields

2. **Analyze Sentiment**
   - Each comment is passed through `TextBlob` and `VADER`
   - Polarity scores (ranging from -1 to +1) are calculated
   - Sentiment labels (Positive, Neutral, Negative) are stored for analysis

3. **Build Recommender**
   - Using `surprise.SVD()`, the model is trained on learner-trainer-rating tuples
   - Model recommends top 5 trainers for any learner

4. **Enhance with Sentiment**
   - Average sentiment per trainer is calculated
   - Recommender results are weighted by sentiment positivity
   - This hybrid method boosts high-performing, well-reviewed trainers

5. **Visualize Trends**
   - Bar charts, histograms, heatmaps and line plots created in `visualization.py`
   - Trends include rating distributions, sentiment counts, and top trainer KPIs

6. **Interactive Dashboard**
   - User inputs learner ID → gets recommendations
   - Trainer dropdown → shows feedback history & average sentiment
   - Sentiment chart auto-updates with filters

---

## 📊 Dashboard Preview

```
📌 Learner ID: [L001] 🔍
📌 Recommendations: Trainer A, Trainer B, Trainer C
📌 Feedback Sentiment: 72% Positive | 18% Neutral | 10% Negative
📈 Charts: Ratings over Time, Trainer Comparison, WordCloud
```

To test it live:
```bash
streamlit run app/streamlit_app.py
```

---

## 📘 Project Goals

✅ Improve training experience using AI and data-driven logic  
✅ Automate feedback analysis via NLP  
✅ Optimize trainer-learner mapping with collaborative filtering  
✅ Deliver end-to-end reproducibility with clean structure & reporting  
✅ Build a deployable dashboard suitable for academic or business use  

---

## 💡 Future Improvements
- Add deep learning sentiment model (e.g. BERT)
- Collect real-time feedback through API endpoints
- Include learner-course domain matching logic
- Use Power BI for alternative executive dashboards

---

## 👨‍🏫 Author
**Jai Dalmotra**   
Email: [jaidalmotra01@gmail.com]  
LinkedIn: [[linkedin.com/in/jai-dalmotra](https://www.linkedin.com/in/jai-dalmotra-64891b1a9/)]  

## License

This project is not open source.  
All content is © 2025 Jai Dalmotra ([@jai-dalmotra](https://github.com/jai-dalmotra)).  
No part of this repository may be copied, reused, or distributed without explicit written permission.

📩 Contact: jaidalmotra01@gmail.com
