"""
Streamlit demo: Adult Census Income - five classifiers, no retraining
"""
import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, confusion_matrix)

from common import (MODEL_COLUMNS, MODEL_FILES, TARGET, clean, encode_target,
                    metrics_from_preds, predict_all)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
PREVIEW_ROWS = 20

# Both "overall" charts share one size and one set of font sizes so they line up
OVERALL_FIG_SIZE = (6, 4.6)
OVERALL_FONTS = {"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
                 "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8}

st.set_page_config(page_title="Income Classifier", page_icon="💼", layout="wide")


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def prepare(df: pd.DataFrame):
    df = clean(df)
    missing = [c for c in MODEL_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {missing}")
        st.stop()
    return df[MODEL_COLUMNS], encode_target(df[TARGET])


def data_key(X: pd.DataFrame, y: pd.Series) -> str:
    """Fingerprint of the loaded data, so results are cleared when the data changes."""
    h = pd.util.hash_pandas_object(X, index=False).sum() + int(y.sum())
    return hashlib.md5(f"{len(X)}-{h}".encode()).hexdigest()


def run_all_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Predict with every saved model (no retraining) and collect the metrics."""
    results = {}
    for name in MODEL_FILES:
        y_pred, y_score = predict_all(load_model(name), X)
        results[name] = {"pred": y_pred, "score": y_score,
                         "metrics": metrics_from_preds(y, y_pred, y_score)}
    return results


def plot_metric_heatmap(table: pd.DataFrame):
    with plt.rc_context(OVERALL_FONTS):
        fig, ax = plt.subplots(figsize=OVERALL_FIG_SIZE)
        sns.heatmap(table, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                    annot_kws={"size": 9}, cbar_kws={"shrink": 0.85})
        ax.set_title("Metric heatmap - all models")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
    return fig


def plot_roc_all(y: pd.Series, results: dict, title: str = "ROC curves - all models"):
    with plt.rc_context(OVERALL_FONTS):
        fig, ax = plt.subplots(figsize=OVERALL_FIG_SIZE)
        for name, r in results.items():
            RocCurveDisplay.from_predictions(y, r["score"], ax=ax, name=name)
        for line in ax.get_lines():
            line.set_linewidth(1.8)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
    return fig


# ---------- sidebar ----------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload test CSV", type="csv",
    help="Must contain the model feature columns and the 'income' column.")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=False)
run_clicked = st.sidebar.button("▶ Run all models on test data", type="primary")
st.sidebar.caption("Models are loaded from disk - nothing is retrained on your upload.")
st.sidebar.title("Results")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))

# ---------- main ----------
st.title("Adult Census Income Classification")
st.subheader("💼 Who earns more than $50K?")
st.write("Evaluate five pre-trained classifiers on the UCI Adult Census Income data.")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
elif use_sample and (ROOT / "test_data.csv").exists():
    raw = pd.read_csv(ROOT / "test_data.csv")
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled sample) to begin.")
    st.stop()

X, y = prepare(raw)
st.success(f"Loaded {len(X):,} rows | {y.mean():.1%} earn >50K")

with st.expander("Preview data", expanded=True):
    shown = min(PREVIEW_ROWS, len(raw))
    st.caption(f"Showing the first {shown} of {len(raw):,} rows "
               f"({raw.shape[1]} columns) of the uploaded test data.")
    st.dataframe(raw.head(PREVIEW_ROWS))

key = data_key(X, y)
if st.session_state.get("run_key") != key:      # new/changed data -> clear old results
    st.session_state.pop("results", None)
    st.session_state["run_key"] = key

if run_clicked:
    with st.spinner("Running all five saved models on the test data..."):
        st.session_state["results"] = run_all_models(X, y)

results = st.session_state.get("results")
if results is None:
    st.info("Press **Run models on test data** in the sidebar to evaluate the "
            "selected model and compare all five models.")
    st.stop()

tab_single, tab_all = st.tabs([f"Selected model: {model_name}", "Overall: all models"])

with tab_single:
    res = results[model_name]
    cols = st.columns(6)
    for col, (k, v) in zip(cols, res["metrics"].items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, res["pred"]),
                               display_labels=["<=50K", ">50K"]).plot(
            ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, res["pred"], target_names=["<=50K", ">50K"],
                                    output_dict=True)
        st.dataframe(pd.DataFrame(rep).T.round(3))

    st.subheader("ROC curve")
    roc_col, _ = st.columns(2)                  # same column width as the overall ROC plot
    with roc_col:
        st.pyplot(plot_roc_all(y, {model_name: res}, title=f"ROC curve - {model_name}"))

with tab_all:
    table = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T.round(4)
    table.index.name = "ML Model Name"
    st.subheader("Metric comparison")
    st.dataframe(table.style.highlight_max(axis=0, color="#c6efce"))
    st.caption("Green = best value per metric on the test data.")

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_metric_heatmap(table))
    with right:
        st.pyplot(plot_roc_all(y, results))

    best = table["MCC"].idxmax()
    st.markdown(f"**Highest MCC on this data:** {best} ({table.loc[best, 'MCC']:.4f})")
