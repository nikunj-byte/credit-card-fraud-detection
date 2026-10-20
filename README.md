# Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using a Random Forest classifier. The project includes a Streamlit web application for batch predictions from CSV files.

## Overview

Credit card fraud detection is an imbalanced classification problem: fraudulent transactions represent only a small fraction of all transactions. A useful model must identify fraud while avoiding excessive false alarms.

This project explores transaction classification and provides an interface for generating fraud-risk predictions.


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

The project uses the Credit Card Fraud Detection dataset from Kaggle.

The dataset contains 284,807 transactions and 30 input features:

- `Time`: elapsed time from the first transaction in the dataset.
- `Amount`: transaction amount.
- `V1`–`V28`: anonymized numerical features.
- `Class`: target label, where `0` represents a legitimate transaction and `1` represents fraud.

The `Class` column is used for evaluation when actual labels are available. It is not used as an input feature for predictions.

## Model

The current model is a Random Forest classifier trained on the transaction features.

The application loads the saved model from `fraud_model.pkl` and the decision threshold from `fraud_threshold.pkl`.

The threshold determines how predicted probabilities are converted into classifications:

- Probability at or above the threshold: potential fraud.
- Probability below the threshold: likely legitimate.

The current threshold is `0.5`. Different thresholds trade off fraud detection (recall) against false alarms (precision).

## Evaluation

The Streamlit dashboard displays these metrics and a graphical confusion matrix when the uploaded CSV contains valid Class labels.

- **Accuracy:** proportion of all transactions classified correctly.
- **Precision:** proportion of transactions flagged as fraud that are actually fraudulent.
- **Recall:** proportion of actual fraudulent transactions detected.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts of true negatives, false positives, false negatives, and true positives.

Evaluation results should be reported using a clearly documented held-out test set. Metrics calculated on training data or data repeatedly used for model selection can overstate generalization performance.

## Installation

Python and pip are required.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Run the application

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Input format

Upload a CSV containing the 30 model features: `Time`, `V1`–`V28`, and `Amount`.

A `Class` column is optional. If present, it is used only to calculate evaluation metrics and is excluded from the model inputs.

## Project structure

```text
fraud-detection/
├── app.py
├── fraud_model.pkl
├── fraud_threshold.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── fraud_detection.ipynb
```

## Limitations

- The dataset's `V1`–`V28` features are anonymized, limiting direct interpretation of their real-world meaning.
- Predictions are estimates, not proof that a transaction is fraudulent.
- The dataset may not represent current transaction patterns or the distribution of fraud in a live payment system.
- The chosen threshold affects the balance between missed fraud and false alarms.
- Metrics from labeled data are only reliable when the evaluation data was not used to train or tune the model.

## Disclaimer

This is an educational machine learning project. It is not a production-ready payment fraud prevention system and should not be used to make real financial decisions without further validation, monitoring, security review, and operational safeguards.
