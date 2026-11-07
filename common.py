"""
Shared helpers for training script and Streamlit app (Adult Census Income)
"""
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

TARGET = "income"
NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain",
           "capital_loss", "hours_per_week"]
CATEGORICAL = ["workclass", "education", "marital_status", "occupation",
               "relationship", "race", "sex", "native_country"]
FEATURES = NUMERIC + CATEGORICAL  # all 14 raw columns of the original dataset

# Columns the models actually use (decided from the EDA in eda.py):
#  - fnlwgt dropped: sampling weight, correlation with income ~ -0.01
#  - education dropped: identical information to education_num (Cramer's V = 1.0)
#  - capital_gain / capital_loss are log1p-transformed (skew 12 / 4.6, >90% zeros)

MODEL_NUMERIC = ["age", "education_num", "hours_per_week"]
MODEL_CAPITAL = ["capital_gain", "capital_loss"]
MODEL_CATEGORICAL = ["workclass", "marital_status", "occupation",
                     "relationship", "race", "sex", "native_country"]
MODEL_COLUMNS = MODEL_NUMERIC + MODEL_CAPITAL + MODEL_CATEGORICAL  # cleaned test CSV columns

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, strip whitespace, turn '?' into NaN."""
    df = df.copy()
    df.columns = [c.strip().replace(".", "_").replace("-", "_") for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()
    return df.replace("?", np.nan)


def encode_target(s: pd.Series) -> pd.Series:
    """'>50K' (or 1) -> 1, '<=50K' (or 0) -> 0."""
    s = s.astype(str).str.strip().str.rstrip(".")
    return s.isin([">50K", "1", "1.0"]).astype(int)


def predict_all(model, X):
    """Return (hard predictions, continuous score used for AUC/ROC)."""
    y_pred = model.predict(X)
    try:
        # works for LR / Tree / KNN / Naive Bayes
        y_score = model.predict_proba(X)[:, 1]
    except (AttributeError, NotFittedError):
        # SVC trained with probability=False
        y_score = model.decision_function(X)
    return y_pred, y_score


def metrics_from_preds(y, y_pred, y_score) -> dict:
    """The six assignment metrics from predictions."""
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def score(model, X, y) -> dict:
    """Compute the six assignment metrics for a fitted pipeline."""
    y_pred, y_score = predict_all(model, X)
    return metrics_from_preds(y, y_pred, y_score)
