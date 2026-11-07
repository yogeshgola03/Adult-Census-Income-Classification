"""
Train 5 classifiers on the Adult Census Income dataset

Usage:  python train_models.py [path/to/adult.csv]

Each classifier is tuned with 5-fold stratified GridSearchCV on the TRAINING split only
(scoring = Matthews Correlation Coefficient, which is robust to class imbalance), then
evaluated once on the held-out test split

class_weight="balanced" is only offered to the SVM, the one model it helps
It lowered the overall quality (MCC) of Logistic Regression and the Decision Tree, so those two are
trained without class weights
KNN and Gaussian Naive Bayes have no class_weight option

Outputs
  model/*.joblib            tuned, fitted pipelines (scaling/encoding included)
  model/metrics.csv         comparison table (Accuracy, AUC, Precision, Recall, F1, MCC)
  model/best_params.json    chosen hyperparameters per model
  model/tuning_summary.csv  CV score, train accuracy, test accuracy per model
  test_data.csv             CLEANED held-out test split: duplicates removed, missing values
                            filled using training-set statistics, unused columns dropped
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (FEATURES, MODEL_CAPITAL, MODEL_CATEGORICAL, MODEL_COLUMNS,
                    MODEL_FILES, MODEL_NUMERIC, TARGET, clean, encode_target, score)

RANDOM_STATE = 42
SVM_SEARCH_ROWS = 8000   # SVM grid search runs on a stratified subsample for speed
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    capital = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, MODEL_NUMERIC),
        ("cap", capital, MODEL_CAPITAL),
        ("cat", categorical, MODEL_CATEGORICAL),
    ])


def search_spaces() -> dict:
    """(classifier, parameter grid) for each model."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [4, 6, 8, 10, 12],
             "clf__min_samples_leaf": [5, 20, 50],
             "clf__criterion": ["gini", "entropy"]}),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 21, 31],
             "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]}),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": np.logspace(-9, 0, 10)}),
        "Support Vector Machine": (
            SVC(kernel="rbf"),
            {"clf__C": [0.5, 1, 3], "clf__gamma": ["scale", 0.01],
             "clf__class_weight": [None, "balanced"]}),
    }


def clean_test_set(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Fill missing test values with TRAINING-set statistics (mode / median)."""
    X_test = X_test.copy()
    n_missing = int(X_test.isna().sum().sum())
    modes = X_train[MODEL_CATEGORICAL].mode().iloc[0]
    medians = X_train[MODEL_NUMERIC + MODEL_CAPITAL].median()
    X_test[MODEL_CATEGORICAL] = X_test[MODEL_CATEGORICAL].fillna(modes)
    X_test[MODEL_NUMERIC + MODEL_CAPITAL] = X_test[MODEL_NUMERIC + MODEL_CAPITAL].fillna(medians)
    print(f"Test set cleaning: filled {n_missing} missing values using training-set statistics")
    return X_test


def main(csv_path: str) -> None:
    df = clean(pd.read_csv(csv_path))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"Columns not found in {csv_path}: {missing}")
    df = df[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    df[TARGET] = encode_target(df[TARGET])
    print(f"Rows: {len(df)}  Raw features: {len(FEATURES)}  "
          f"Positive rate (>50K): {df[TARGET].mean():.3f}")

    X, y = df[MODEL_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # cleaned test data: what the app uploads and evaluates
    X_test = clean_test_set(X_train, X_test)
    MODEL_DIR.mkdir(exist_ok=True)
    test_out = X_test.copy()
    test_out[TARGET] = y_test.map({0: "<=50K", 1: ">50K"})
    test_out.to_csv(ROOT / "test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mcc = make_scorer(matthews_corrcoef)

    rows, best_params, summary = {}, {}, {}
    for name, (clf, grid) in search_spaces().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring=mcc, cv=cv, n_jobs=-1, refit=True)
        t0 = time.time()
        if name == "Support Vector Machine":
            idx = X_train.sample(SVM_SEARCH_ROWS, random_state=RANDOM_STATE).index
            search.fit(X_train.loc[idx], y_train.loc[idx])
            model = search.best_estimator_  # best params found on the subsample ...
            model.fit(X_train, y_train)     # ... then refit on all training rows
        else:
            search.fit(X_train, y_train)
            model = search.best_estimator_
        params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                  for k, v in search.best_params_.items()}
        best_params[name] = params
        rows[name] = score(model, X_test, y_test)
        summary[name] = {"CV MCC (train)": round(search.best_score_, 4),
                         "Train accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
                         "Test accuracy": round(rows[name]["Accuracy"], 4)}
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        print(f"{name:<24} {time.time() - t0:6.1f}s  best={params}")

    metrics = pd.DataFrame(rows).T.round(4)
    metrics.index.name = "ML Model Name"
    metrics.to_csv(MODEL_DIR / "metrics.csv")
    pd.DataFrame(summary).T.rename_axis("ML Model Name").to_csv(MODEL_DIR / "tuning_summary.csv")
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))
    print("\n" + metrics.to_string())
    print("\n" + pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "adult.csv")
