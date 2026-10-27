
# AI-Powered Customer Churn Prediction & Customer Analytics Platform

## 📌 Project Overview
This project uses Machine Learning to predict whether a customer is likely to leave a telecom company. It also analyzes customer data to identify patterns and factors related to customer churn.

## 🎯 Objectives
- Predict customer churn using Machine Learning.
- Analyze customer behavior and subscription patterns.
- Identify factors that may lead to customer churn.
- Visualize important insights using charts.
- Help businesses improve customer retention.

## 📊 Dataset
The project uses the IBM Telco Customer Churn dataset.

- **Original dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Cleaned dataset:** `telco_churn_cleaned.csv`
- **Target column:** `Churn`
- **Prediction classes:** Yes (customer left) and No (customer stayed)

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Machine Learning

## ⚙️ Project Workflow
1. Load the customer dataset.
2. Clean missing values and duplicate records.
3. Perform Exploratory Data Analysis (EDA).
4. Visualize customer churn patterns.
5. Prepare features for Machine Learning.
6. Train a classification model.
7. Evaluate model performance.
8. Predict customer churn.

## 📈 Exploratory Data Analysis
The project explores:
- Overall customer churn distribution.
- Contract type and customer churn.
- Customer tenure and churn.
- Customer behavior patterns.

## 🤖 Machine Learning
A classification model will be trained to predict customer churn. Model performance will be evaluated using suitable metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

## ▶️ How to Run the Project

### 1. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Python application
```bash
python app.py
```

Make sure the original dataset is present in the project folder.

## 📁 Project Files
- `app.py` — Main Python application.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — Original dataset.
- `telco_churn_cleaned.csv` — Cleaned dataset.
- `churn_distribution.png` — Churn distribution chart.
- `contract_vs_churn.png` — Contract type analysis.
- `tenure_vs_churn.png` — Tenure analysis.

## 🚀 Future Scope
- Build an interactive Streamlit dashboard.
- Compare multiple Machine Learning algorithms.
- Improve prediction performance.
- Provide customer retention insights.

## 👨‍💻 Author
BCA (Artificial Intelligence & Machine Learning) Student
