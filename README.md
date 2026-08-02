# Tourism Package Purchase Prediction using MLOps

## Project Overview

This project predicts whether a customer is likely to purchase a tourism package using Machine Learning. The project demonstrates a complete end-to-end MLOps pipeline including data preprocessing, model training, experiment tracking, deployment, and CI/CD automation.

---

## Problem Statement

The objective is to build a classification model that predicts whether a customer will purchase a tourism package (`ProdTaken`) based on demographic and behavioural attributes.

---

## Project Structure

```
tourism-mlops-project/
│
├── requirements.txt
├── README.md
│
├── tourism_project/
│   ├── artifacts/
│   ├── data/
│   │   └── tourism.csv
│   ├── deployment/
│   │   ├── app.py
│   │   ├── model.pkl
│   │   ├── feature_columns.pkl
│   │   └── requirements.txt
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   └── train.py
│
└── .github/
    └── workflows/
        └── mlops_pipeline.yml
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Joblib
- Streamlit
- GitHub Actions

---

## Machine Learning Models

The following classification algorithms were trained and compared:

- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

Hyperparameter tuning was performed using GridSearchCV.

---

## Best Performing Model

- **Algorithm:** Random Forest
- **Accuracy:** ~90.8%

---

## Features

- Data preprocessing
- Missing value handling
- Label encoding
- Train-test split
- Hyperparameter tuning
- MLflow experiment tracking
- Automatic best model selection
- Model serialization
- Streamlit deployment
- GitHub Actions CI/CD pipeline

---

## Deployment

The trained model is deployed using Streamlit.

Run locally using:

```bash
streamlit run tourism_project/deployment/app.py
```

---

## Workflow

1. Register dataset
2. Preprocess data
3. Train multiple ML models
4. Tune hyperparameters
5. Track experiments with MLflow
6. Save the best model
7. Deploy using Streamlit
8. Automate using GitHub Actions

---

## Author

Sushant Bose
