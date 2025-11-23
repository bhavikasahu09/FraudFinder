# FraudFinder: Predictive Modeling for Insurance Claim Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## 📌 Overview

FraudFinder uses machine learning to automatically identify fraudulent insurance claims by analyzing claim patterns and customer data. It provides real-time fraud predictions through an interactive web dashboard with explainability features.

**Domain:** BFSI (Banking, Financial Services, Insurance)  
**Type:** Supervised Classification  
**Institution:** CDAC Pune - DBDA Program

---

## 🎯 Features

- **Fraud Detection:** Binary classification (Fraudulent/Genuine) with confidence scores
- **Multiple ML Models:** Logistic Regression, Random Forest, XGBoost, Neural Networks
- **Explainability:** SHAP values showing why claims are flagged
- **Interactive Dashboard:** User-friendly web interface for real-time predictions
- **Visualization:** Fraud patterns, feature importance, and performance metrics
- **Model Comparison:** Side-by-side evaluation of different algorithms

---

## 🛠️ Tech Stack

**Languages & Libraries:**
- Python 3.8+
- pandas, numpy (Data processing)
- scikit-learn, xgboost (Machine Learning)
- imbalanced-learn (Handling class imbalance)
- SHAP (Model explainability)
- matplotlib, seaborn, plotly (Visualization)

**Deployment:**
- Streamlit (Web framework)
- Streamlit Cloud (Hosting)

**Tools:**
- Jupyter Notebook (Development)
- Git/GitHub (Version control)

---

## 📂 Project Structure

```
insurance-fraud-detection/
│
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned datasets
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── models/
│   ├── best_model.pkl          # Saved trained model
│   └── scaler.pkl              # Feature scaler
│
├── app/
│   ├── app.py                  # Streamlit dashboard
│   └── utils.py                # Helper functions
│
├── docs/
│   ├── project_report.pdf
│   └── presentation.pptx
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone Repository
```bash
git clone https://github.com/sumitnagpure/FraudFinder.git
cd fraudfinder
```

### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Dataset
1. Go to [Kaggle Insurance Fraud Dataset](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
2. Download and place in `data/raw/` folder

---

## 💻 Usage

### 1. Data Preparation
```bash
jupyter notebook notebooks/01_data_cleaning.ipynb
```

### 2. Exploratory Data Analysis
```bash
jupyter notebook notebooks/02_eda.ipynb
```

### 3. Model Training
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 4. Run Streamlit Dashboard
```bash
streamlit run app/app.py
```

Dashboard will open at `http://localhost:8501`

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD |

*Results will be updated after model training*

---

## 📈 Project Roadmap

- [x] Project setup and repository initialization
- [ ] **Week 1:** Dataset collection, cleaning, EDA
- [ ] **Week 2:** Model training, tuning, evaluation
- [ ] **Week 3:** SHAP explainability, dashboard development
- [ ] **Week 4:** Deployment, testing, documentation

---

## 👥 Team

**CDAC Pune - DBDA Batch 2024-25**

- **Member 1** - AI/ML Focus - Model Development
- **Member 2** - ECE Background - Data Engineering
- **Member 3** - Gaming Background - UI/Visualization

---

## 📝 Key Learnings

- Handling imbalanced datasets in classification problems
- Implementing explainable AI using SHAP
- End-to-end ML pipeline: data → model → deployment
- Real-world BFSI domain application

---

## 🔮 Future Enhancements

- [ ] Add deep learning models (LSTM, Transformers)
- [ ] Implement real-time API endpoint
- [ ] Add user authentication
- [ ] Deploy on AWS/Azure with database integration
- [ ] Mobile app version

---

## 📄 License

This project is for educational purposes as part of CDAC DBDA curriculum.

---

## 🙏 Acknowledgments

- CDAC Pune for project guidance
- Kaggle for datasets
- Open-source ML community

---

## 📧 Contact

For queries or collaboration:
- GitHub Issues: [Open an issue](https://github.com/your-username/fraudfinder/issues)
- Email: your-email@example.com

---

**⭐ If you find this project useful, please consider giving it a star!**