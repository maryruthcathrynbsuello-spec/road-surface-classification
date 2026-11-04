import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures"
REPORT_DIR = ROOT / "docs" / "data_reports"

MODEL_NAME = "efficientnetb0_v2"
METRICS_FILE = REPORT_DIR / f"{MODEL_NAME}_metrics.json"


HISTORY_FILE = Path(__file__).with_name("history.json")


def pretty(name):
    return str(name).replace("_", " ")


def find_figure(*patterns):
    for pattern in patterns:
        matches = sorted(FIG_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_metrics():
    if not METRICS_FILE.exists():
        return None
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        st.error(f"Could not read {METRICS_FILE.name}: {e}")
        return None


def load_history():
    """Return saved upload records, or None if no predictions are available."""
    if not HISTORY_FILE.exists():
        return None
    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        st.error(f"Could not read {HISTORY_FILE.name}: {e}")
        return None
    if not isinstance(records, list):
        st.error(f"Expected {HISTORY_FILE.name} to contain a list of records.")
        return None
    return records or None


metrics = load_metrics()

# Header
st.title("Road Surface Condition Dashboard")
st.caption("Overview of the model and of the images classified so far.")

# Model headline numbers
st.subheader("Model Results (Test Set)")
m1, m2, m3 = st.columns(3)
if metrics:
    per_class = metrics.get("per_class", {})
    m1.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
    m2.metric("Test Macro F1", f"{metrics['test_macro_f1']:.2f}")
    m3.metric("Test Images", f"{sum(v.get('support', 0) for v in per_class.values()):,}")
    st.caption(f"Model: {MODEL_NAME}. See the Model Performance page for details.")
else:
    m1.metric("Test Accuracy", "—")
    m2.metric("Test Macro F1", "—")
    m3.metric("Test Images", "—")
    st.caption(f"No results yet. These fill in once "
               f"`docs/data_reports/{METRICS_FILE.name}` is added.")

# Model figures
st.subheader("Model Figures")
f1, f2 = st.columns(2)
with f1:
    cm_path = find_figure(f"{MODEL_NAME}_test_confusion*.png", "*test_confusion*.png")
    if cm_path:
        st.markdown("**Confusion matrix (test)**")
        st.image(str(cm_path))
    else:
        st.info("Confusion matrix not found in `docs/figures/`.")
with f2:
    curves = find_figure(f"{MODEL_NAME}*curve*.png", f"{MODEL_NAME}*history*.png")
    if curves:
        st.markdown("**Learning curves**")
        st.image(str(curves))
    else:
        st.info("Learning curves not found in `docs/figures/`.")

# Prediction history
st.subheader("Prediction History")
history = load_history()
if not history:
    h1, h2 = st.columns(2)
    h1.metric("Total Predictions", 0)
    h2.metric("Classes Seen", 0)
    st.info("No predictions yet. Class counts will appear here once images "
            "have been classified.")
else:
    labels = [
        pretty(label)
        for record in history
        for label in record.get("labels", [])
    ]
    counts = pd.Series(labels, dtype="string").value_counts()
    h1, h2 = st.columns(2)
    h1.metric("Total Predictions", f"{len(labels):,}")
    h2.metric("Classes Seen", len(counts))
    st.markdown("**Predictions per class**")
    st.bar_chart(counts)