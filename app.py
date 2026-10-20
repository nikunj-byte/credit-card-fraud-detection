
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load the trained model and threshold
model = joblib.load("fraud_model.pkl")
threshold = joblib.load("fraud_threshold.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Upload transaction data to estimate fraud risk.")

st.info(
    "Upload a CSV containing the 30 model features: "
    "Time, V1–V28 and Amount. A Class column is optional."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("The uploaded CSV contains no transactions.")

        else:
            # Separate features from the optional target label
            x_data = data.drop(columns=["Class"], errors="ignore")
            required_features = list(model.feature_names_in_)

            missing = [
                col for col in required_features
                if col not in x_data.columns
            ]

            if missing:
                st.error(f"Missing required columns: {missing}")

            else:
                # Match the feature order used during training
                x_data = x_data[required_features]

                # Convert features to numeric values
                x_data = x_data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if x_data.isnull().any().any():
                    st.error(
                        "The CSV contains missing or non-numeric "
                        "feature values. Please correct the file."
                    )

                elif not np.isfinite(x_data.to_numpy()).all():
                    st.error(
                        "The CSV contains infinite values. "
                        "Please correct the file."
                    )

                else:
                    # Generate predictions
                    probabilities = model.predict_proba(x_data)[:, 1]
                    predicted = (
                        probabilities >= threshold
                    ).astype(int)

                    results = data.copy()
                    results["Fraud Probability"] = probabilities
                    results["Prediction"] = [
                        "Potential fraud" if p == 1
                        else "Likely legitimate"
                        for p in predicted
                    ]

                    # Prediction summary
                    st.subheader("Prediction Summary")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Transactions", len(results))
                    col2.metric(
                        "Flagged as fraud",
                        int(predicted.sum())
                    )
                    col3.metric(
                        "Likely legitimate",
                        int((predicted == 0).sum())
                    )

                    # Evaluation dashboard: only when actual labels exist
                    if "Class" in data.columns:
                        actual = pd.to_numeric(
                            data["Class"],
                            errors="coerce"
                        )

                        if (
                            actual.notna().all()
                            and actual.isin([0, 1]).all()
                        ):
                            actual = actual.astype(int)

                            st.header("📊 Model Evaluation Dashboard")

                            col1, col2, col3, col4 = st.columns(4)

                            col1.metric(
                                "Accuracy",
                                f"{accuracy_score(actual, predicted):.2%}"
                            )
                            col2.metric(
                                "Fraud Precision",
                                f"{precision_score(actual, predicted, zero_division=0):.2%}"
                            )
                            col3.metric(
                                "Fraud Recall",
                                f"{recall_score(actual, predicted, zero_division=0):.2%}"
                            )
                            col4.metric(
                                "Fraud F1-score",
                                f"{f1_score(actual, predicted, zero_division=0):.3f}"
                            )

                            st.subheader("Confusion Matrix")

                            cm = confusion_matrix(
                                actual,
                                predicted,
                                labels=[0, 1]
                            )

                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.imshow(cm, cmap="Blues")

                            ax.set_xticks([0, 1])
                            ax.set_yticks([0, 1])
                            ax.set_xticklabels(
                                ["Legitimate", "Fraud"]
                            )
                            ax.set_yticklabels(
                                ["Legitimate", "Fraud"]
                            )

                            ax.set_xlabel("Predicted class")
                            ax.set_ylabel("Actual class")
                            ax.set_title("Actual vs Predicted")

                            for i in range(2):
                                for j in range(2):
                                    ax.text(
                                        j, i, str(cm[i, j]),
                                        ha="center",
                                        va="center",
                                        color="black"
                                    )

                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        else:
                            st.warning(
                                "Predictions were generated, but evaluation "
                                "was skipped because Class must contain "
                                "only valid 0 and 1 labels."
                            )

                    # Transaction results
                    st.subheader("Transaction Results")
                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # Download predictions
                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download predictions CSV",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

    except pd.errors.EmptyDataError:
        st.error(
            "The uploaded file is empty or has no readable CSV data."
        )

    except Exception as e:
        st.error(f"Could not process the CSV: {e}")
