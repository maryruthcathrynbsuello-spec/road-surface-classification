import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.results_loader import (  
    HISTORY_FILE, MODEL_NAME, comparison_table, find_figure, load_prediction_history,
    metrics_path, model_label, pretty, summary,
)

try:
    st.set_page_config(page_title="Dashboard", layout="wide")
except Exception:
    pass  

main = summary(metrics_path(), "test")
chosen_label = model_label(metrics_path().stem)


st.title("Road Surface Condition Dashboard")
st.caption("Overview of the model and of the images classified so far.")

if main["error"]:
    st.error(main["error"])


split = main["label"]                     
split_word = f"{split} " if split else ""
pc = main["per_class"]
shown = pc.dropna(axis=1, how="all") if pc is not None else None

with st.container(border=True):
    title = f"#### Model Results ({split} Set) · {chosen_label}" if split \
        else f"#### Model Results · {chosen_label}"
    st.markdown(title)
    c1, c2, c3, c4, c5 = st.columns(5)
    best = weak = None
    if shown is not None and "F1" in shown.columns and shown["F1"].notna().any():
        best, weak = shown["F1"].idxmax(), shown["F1"].idxmin()

    c1.metric(f"{split_word}Accuracy",
              f"{main['accuracy']:.1%}" if main["accuracy"] is not None else "—")
    c2.metric(f"{split_word}Macro F1",
              f"{main['macro_f1']:.2f}" if main["macro_f1"] is not None else "—")
    if main["total_images"] is not None:
        c3.metric("Images", f"{main['total_images']:,}")
    else:
        c3.metric("Epochs Trained",
                  main["epochs_total"] if main["epochs_total"] is not None else "—")
    c4.metric("Best Class (F1)", best.title() if best else "—")
    c5.metric("Weakest Class (F1)", weak.title() if weak else "—")
    if main["data"] is None:
        st.caption(f"No results yet. These fill in once "
                   f"`docs/data_reports/{metrics_path().name}` is available.")
    elif split == "Validation":
        st.caption("Validation scores. The test set is kept for one final evaluation.")

st.subheader("Per-Class Scores")
if shown is not None:
    score_cols = [c for c in ["Precision", "Recall", "F1"] if c in shown.columns]
    config = {
        "Test images": st.column_config.NumberColumn("Images", format="%d"),
        "Precision": st.column_config.ProgressColumn(
            "Precision", min_value=0.0, max_value=1.0, format="%.2f"),
        "Recall": st.column_config.ProgressColumn(
            "Recall", min_value=0.0, max_value=1.0, format="%.2f"),
        "F1": st.column_config.ProgressColumn(
            "F1", min_value=0.0, max_value=1.0, format="%.2f"),
    }
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("**Scores per class**")
            st.bar_chart(shown[score_cols])
    with right:
        with st.container(border=True):
            st.markdown("**Table**")
            st.dataframe(shown, column_config={k: v for k, v in config.items()
                                               if k in shown.columns})
else:
    st.info("Per-class scores are not available yet.")

st.subheader("Model Figures")
f1, f2 = st.columns(2)
with f1:
    with st.container(border=True):
        cm_path = find_figure(f"{MODEL_NAME}_test_confusion*.png", "*test_confusion*.png")
        st.markdown("**Confusion matrix (test)**")
        if cm_path:
            st.image(str(cm_path))
        else:
            st.info("Confusion matrix not found in `docs/figures/`.")
with f2:
    with st.container(border=True):
        curves = find_figure(f"{MODEL_NAME}*curve*.png", f"{MODEL_NAME}*history*.png")
        st.markdown("**Learning curves**")
        if curves:
            st.image(str(curves))
        else:
            st.info("Learning curves not found in `docs/figures/`.")

comp = comparison_table()
if comp is not None:
    st.subheader("Model Comparison")
    with st.container(border=True):
        st.dataframe(comp, column_config={
            "Accuracy": st.column_config.ProgressColumn(
                "Accuracy", min_value=0.0, max_value=1.0, format="%.2f"),
            "Macro F1": st.column_config.ProgressColumn(
                "Macro F1", min_value=0.0, max_value=1.0, format="%.2f"),
        })

st.subheader("Prediction History")
history, class_col, hist_err = load_prediction_history()
with st.container(border=True):
    if hist_err:
        st.error(hist_err)
    if history is None:
        h1, h2 = st.columns(2)
        h1.metric("Total Predictions", 0)
        h2.metric("Classes Seen", 0)
        st.info("No predictions yet. Class counts will appear here once images "
                "have been classified.")
    else:
        h1, h2 = st.columns(2)
        h1.metric("Total Predictions", f"{len(history):,}")
        if class_col is None:
            h2.metric("Classes Seen", "—")
            st.warning("The history file has no class column I recognise. "
                       f"Columns found: {', '.join(map(str, history.columns))}.")
        else:
            counts = history[class_col].value_counts()
            counts.index = [pretty(c) for c in counts.index]
            h2.metric("Classes Seen", len(counts))
            st.markdown("**Predictions per class**")
            st.bar_chart(counts)