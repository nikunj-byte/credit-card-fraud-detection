# Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using a Random Forest classifier. The project includes a Streamlit web application for batch predictions from CSV files.

## Overview

Credit card fraud detection is an imbalanced classification problem: fraudulent transactions represent only a small fraction of all transactions. A useful model must identify fraud while avoiding excessive false alarms.

This project uses a Random Forest classifier and a Streamlit interface to generate fraud-risk predictions, evaluate performance on labeled data, and download prediction results.

## Features

- Upload transaction data in CSV format.
- Validate required features and detect missing, non-numeric, or infinite values.
- Estimate fraud probabilities for each transaction.
- Classify transactions using a configurable decision threshold.
- Display transaction counts and prediction summaries.
- Show an evaluation dashboard when valid ground-truth labels are available.
- Display accuracy, fraud precision, fraud recall, and F1-score.
- Visualize the confusion matrix using Matplotlib.
- Download predictions and probabilities as a CSV file.

## Dataset

The project uses the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

The dataset contains 284,807 transactions and 30 input features, along with a target column:

- `Time`: elapsed time from the first transaction in the dataset.
- `Amount`: transaction amount.
- `V1`–`V28`: anonymized numerical features obtained through Principal Component Analysis (PCA).
- `Class`: target label, where `0` represents a legitimate transaction and `1` represents fraud.

The dataset contains 492 fraudulent transactions, approximately 0.173% of all transactions.

The `Class` column is used for evaluation when actual labels are available. It is excluded from the model inputs.

## Model

The project uses a **Random Forest classifier** trained on transaction features.

The application loads the trained model from `fraud_model.pkl` and the classification threshold from `fraud_threshold.pkl`.

The selected classification threshold is **0.35**, chosen using the validation set to maximize the F1-score.

- Predicted fraud probability greater than or equal to `0.35`: potential fraud.
- Predicted fraud probability below `0.35`: likely legitimate.

The threshold controls the trade-off between fraud detection (recall) and false alarms (precision).

## Evaluation

The selected Random Forest model was evaluated on a held-out test set that was not used to train the model or select the threshold.

### Final Test Results

| Metric | Score |
|---|---:|
| Average Precision | 0.8682 |
| Precision | 89.25% |
| Recall | 84.69% |
| F1-score | 0.8691 |

### Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,854 | 10 |
| Actual Fraud | 15 | 83 |

On the held-out test set, the model correctly identified 83 of 98 fraudulent transactions and missed 15. It incorrectly flagged 10 legitimate transactions as fraudulent.

### Evaluation Metrics

- **Average Precision:** summarizes precision-recall performance across classification thresholds.
- **Precision:** proportion of transactions flagged as fraud that were actually fraudulent.
- **Recall:**