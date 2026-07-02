# 🏏 Cricket Analytics using Machine Learning

> End-to-End Data Analytics & Machine Learning Pipeline for Predicting ODI Cricket Performance

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## 📌 Overview

This project presents an end-to-end machine learning pipeline for predicting ODI cricket player performance using statistical modeling and supervised machine learning.

Developed as part of my **M.Sc. Statistics dissertation at Babasaheb Bhimrao Ambedkar University (BBAU), Lucknow**, the project demonstrates the complete data science workflow—from raw data preprocessing to predictive modeling and evaluation.

The pipeline predicts:

- 🏏 Batting Runs
- 🎯 Bowling Wickets

using historical ODI performance and engineered temporal features.

---

# 🚀 Objectives

- Build a complete analytics pipeline for ODI cricket data.
- Perform data cleaning and preprocessing.
- Engineer predictive time-series features.
- Compare multiple machine learning algorithms.
- Evaluate model performance using statistical metrics.
- Prevent data leakage through proper temporal validation.

---

# 📊 Dataset

The project uses historical ODI cricket match data.

### Batting Dataset

- 303 ODI innings
- Match-wise batting records
- Runs
- Balls Faced
- Strike Rate
- Opposition
- Venue
- Match Date

### Bowling Dataset

- 88 ODI bowling records
- Wickets
- Economy
- Overs
- Opposition
- Venue
- Match Date

---

# ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- Missing value treatment
- Duplicate removal
- Date conversion
- Chronological sorting
- Feature encoding
- Outlier inspection
- Data type optimization

---

# 🧠 Feature Engineering

More than **20 temporal features** were created, including:

- Lag Features (Lag-1, Lag-2, Lag-3)
- Rolling Mean (3, 5, 10 matches)
- Exponential Moving Average (EMA)
- Form Index
- Opponent Statistics
- Venue Performance
- Boundary Percentage
- Momentum Features
- Home/Away Indicator
- Strike Rate Trend

To eliminate **data leakage**, all rolling statistics were generated using `shift(1)` before aggregation.

---

# 🤖 Machine Learning Models

The following models were benchmarked:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- LightGBM
- SARIMAX
- Stacking Ensemble

---

# 📈 Model Evaluation

Evaluation metrics include:

- RMSE
- MAE
- R² Score
- MAPE
- Residual Diagnostics
- Bootstrap Confidence Intervals

---

# 📌 Results

### Batting Prediction

- Best R² = **0.987**
- RMSE = **5.4 Runs**

### Bowling Prediction

- Best RMSE = **1.39 Wickets**
- Best Model = **Stacking Ensemble**

The project also validates statistical assumptions using:

- Variance Inflation Factor (VIF)
- Residual Analysis
- Q-Q Plot
- Bootstrap Confidence Intervals

---

# 📊 Visualizations

The project includes:

- Feature Importance
- Residual Plots
- Correlation Heatmaps
- Q-Q Plots
- Prediction vs Actual
- Time Series Trends

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# 📂 Project Structure

```
Cricket-Analytics/
│
├── data/
│   ├── batting.csv
│   ├── bowling.csv
│
├── notebooks/
│   ├── Batting_Pipeline.ipynb
│   ├── Bowling_Pipeline.ipynb
│
├── models/
│
├── images/
│
├── results/
│
├── requirements.txt
│
└── README.md
```

---

# 📈 Workflow

```
Raw Data
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Comparison
      │
      ▼
Performance Evaluation
      │
      ▼
Prediction & Visualization
```

---

# 💡 Key Learnings

- End-to-end machine learning workflow
- Time-series feature engineering
- Data leakage prevention
- Statistical model diagnostics
- Model comparison and evaluation
- Predictive analytics using Python

---

# 🚀 Future Improvements

- Deploy using Streamlit
- Live Cricbuzz/API integration
- XGBoost & CatBoost implementation
- Hyperparameter Optimization
- Docker deployment
- Automated data pipeline

---

# 👨‍💻 Author

**Ekeshwar Singh**

M.Sc. Statistics  
Babasaheb Bhimrao Ambedkar University, Lucknow

📧 Email: singhekesh11011@gmail.com

🔗 LinkedIn: *(Add your LinkedIn URL)*

🐙 GitHub: *(Add your GitHub Profile URL)*

---

## ⭐ If you found this project useful, please consider giving it a star!
