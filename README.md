# 💳 Credit Card Fraud Detection — End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

---

## 📌 Project Overview

A production-quality **end-to-end machine learning pipeline** for detecting fraudulent credit card transactions. Built with professional standards including full exploratory data analysis, feature engineering, class imbalance handling, model training, evaluation, and improvement.

This project demonstrates real-world data science practices such as:
- Handling **severe class imbalance** (only 1.51% fraud)
- Applying **SMOTE** for synthetic oversampling
- **Threshold tuning** to optimize for Recall over Accuracy
- Building **interpretable models** suitable for business stakeholders

---

## 🎯 Problem Statement

Credit card fraud costs the global economy **billions of dollars annually**. Traditional rule-based systems miss subtle patterns. This project builds a machine learning model that:

- Automatically scores every transaction with a **fraud probability (0–100%)**
- Prioritizes **Recall** — catching real fraud matters more than avoiding false alarms
- Provides **interpretable insights** into what drives fraud

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── 📓 notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb   # Main Jupyter notebook (all 8 phases)
│
├── 📊 data/
│   └── credit_card_fraud_10k.csv           # Dataset (10,000 transactions)
│
├── 🖼️ images/
│   └── (visualizations auto-saved when notebook runs)
│
├── 📄 requirements.txt                     # Python dependencies
├── 📄 .gitignore                           # Files excluded from Git
└── 📄 README.md                            # This file
```

---

## 📊 Dataset Description

| Column | Type | Description |
|---|---|---|
| `transaction_id` | ID | Unique transaction identifier (dropped before modeling) |
| `amount` | Numerical | Dollar value of the transaction |
| `transaction_hour` | Numerical | Hour of day (0–23) |
| `merchant_category` | Categorical | Type of merchant: Food, Travel, Grocery, Clothing, Electronics |
| `foreign_transaction` | Binary | 1 = transaction made abroad |
| `location_mismatch` | Binary | 1 = billing address doesn't match transaction location |
| `device_trust_score` | Numerical | Device trustworthiness score (25–99) |
| `velocity_last_24h` | Numerical | Number of transactions in the last 24 hours |
| `cardholder_age` | Numerical | Age of the cardholder (18–69) |
| `is_fraud` | **Target** | 1 = Fraud, 0 = Legitimate |

**Class Distribution:** 98.49% Legitimate | 1.51% Fraud (severely imbalanced)

---

## 🔬 Project Phases

### Phase 1 — Data Understanding
- Loaded and explored 10,000 transactions
- Identified severe class imbalance (1.51% fraud)
- Mapped each column to its business meaning

### Phase 2 — Data Cleaning
- Verified: no missing values, no duplicates, no type errors
- Retained outliers in `amount` (real fraud signals, not noise)
- Dropped `transaction_id` (identifier, not a feature)

### Phase 3 & 4 — EDA & Visualizations
- 10+ professional visualizations (histograms, box plots, heatmaps, pair plots)
- Identified top fraud signals: `location_mismatch`, `foreign_transaction`, `velocity_last_24h`
- Discovered merchant category fraud patterns (Grocery highest, Electronics lowest)

### Phase 5 — Feature Engineering & Preprocessing
- Created 4 new risk-based features from domain knowledge
- One-hot encoded `merchant_category`
- Applied `StandardScaler` (fit on train only — no leakage)
- Applied **SMOTE** on training set → balanced from 1.51% to 50% fraud

### Phase 6 — Logistic Regression Model
- Trained baseline model (no SMOTE) — high accuracy, zero fraud recall
- Trained SMOTE model — dramatically improved fraud recall
- Full evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

### Phase 7 — Model Improvement
- **Threshold tuning**: tested 0.10 → 0.90 to find optimal fraud cutoff
- **5-fold Stratified Cross-Validation** inside a full pipeline
- **Regularization search**: tested 7 values of C parameter

### Phase 8 — Insights & Conclusion
- Actionable business fraud patterns
- Model strengths, weaknesses, and future improvement roadmap
- Saved model artifacts (`.pkl` files)

---

## 📈 Key Results

| Metric | Baseline (No SMOTE) | Final Model (Tuned) |
|---|---|---|
| Accuracy | ~98.5% | Lower (expected) |
| **Recall (Fraud)** | **Very Low** | **Significantly Higher** |
| F1-Score (Fraud) | Very Low | Improved |
| ROC-AUC | Moderate | Better |

> ⚠️ Accuracy is misleading for imbalanced fraud datasets. **Recall** is the primary metric.

---

## 💡 Key Business Insights

- 🔴 **Foreign transactions** → significantly elevated fraud rate → require extra verification
- 🔴 **Location mismatch** → strongest single fraud predictor → auto-flag for review
- 🟠 **High velocity** (5+ transactions/24h) → fraud pattern → velocity limits recommended
- 🟠 **Low device trust** → new/unfamiliar device → trigger MFA
- 🟡 **Grocery category** → highest merchant fraud rate (small unnoticed charges)

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended — No Setup)
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Credit_Card_Fraud_Detection.ipynb`
3. Upload `credit_card_fraud_10k.csv` via the sidebar Files panel
4. Click **Runtime → Run All**

### Option 2: Local Environment
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook notebooks/Credit_Card_Fraud_Detection.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML model, preprocessing, evaluation |
| Imbalanced-learn | SMOTE for class balancing |
| Jupyter Notebook | Interactive development |

---

## 🔮 Recommended Next Steps

- **Random Forest / XGBoost** — captures non-linear patterns (likely better recall)
- **Isolation Forest** — anomaly detection without labels
- **Neural Network (MLP)** — scales well to 100k+ transactions
- **Real-time API** — deploy model as a REST endpoint for live scoring
- **Model monitoring** — track performance drift as fraud patterns evolve

---

## 🤝 Connect

Built by **[Your Name]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
