import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.results_loader import (  # noqa: E402
    FIG_DIR, MODEL_NAME, comparison_table, find_figure, metrics_path,
    model_label, setup_text, summary,
)

try:
    st.set_page_config(page_title="Model Performance", layout="wide")
except Exception:
    pass  


def waiting(what, hint):
    st.info(f"**{what}** is not available yet. {hint}")


def score_config(columns):
    config = {
        "Test images": st.column_config.NumberColumn("Test images", format="%d"),
        "Precision": st.column_config.ProgressColumn(
            "Precision", min_value=0.0, max_value=1.0, format="%.2f"),
        "Recall": st.column_config.ProgressColumn(
            "Recall", min_value=0.0, max_value=1.0, format="%.2f"),
        "F1": st.column_config.ProgressColumn(
            "F1", min_value=0.0, max_value=1.0, format="%.2f"),
    }
    return {k: v for k, v in config.items() if k in columns}


MAIN_FILE = metrics_path()
main = summary(MAIN_FILE, "test")
chosen_label = model_label(MAIN_FILE.stem)
split = main["label"]                      
split_word = f"{split} " if split else ""

st.title("Model Performance")
st.caption(f"Results of the chosen model ({chosen_label}).")

if main["error"]:
    st.error(main["error"])
elif main["data"] is None:
    st.warning(f"Metrics file not found: `docs/data_reports/{MAIN_FILE.name}`.")
elif main["accuracy"] is None or main["macro_f1"] is None:
    st.warning("The metrics file was found, but accuracy or macro F1 could not "
               "be read from it.")
    with st.expander("Show the file contents"):
        st.json(main["data"], expanded=1)

st.subheader(f"1. Headline Numbers ({split} Set)" if split else "1. Headline Numbers")
h1, h2, h3 = st.columns(3)
h1.metric(f"{split_word}Accuracy",
          f"{main['accuracy']:.1%}" if main["accuracy"] is not None else "—")
h2.metric(f"{split_word}Macro F1",
          f"{main['macro_f1']:.2f}" if main["macro_f1"] is not None else "—")
if main["total_images"] is not None:
    h3.metric("Images", f"{main['total_images']:,}")
elif main["epochs_total"] is not None:
    h3.metric("Epochs Trained", main["epochs_total"])
else:
    h3.metric("Epochs Trained", "—")

if split == "Validation":
    st.caption("These are validation scores, measured on images held out during "
               "training. The test set is kept for one final evaluation.")
elif split == "Test":
    st.caption("These numbers come from the test set: images the model never "
               "saw during training or tuning.")

st.subheader("2. Per-Class Results")
pc = main["per_class"]
if pc is not None:
    shown = pc.dropna(axis=1, how="all")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(shown, column_config=score_config(shown.columns))
    with right:
        score_cols = [c for c in ["Precision", "Recall", "F1"] if c in shown.columns]
        st.bar_chart(shown[score_cols])
    if "F1" in shown.columns:
        weak = shown["F1"].idxmin()
        st.caption(f"Weakest class by F1: **{weak}** "
                   f"({shown['F1'].min():.2f}).")
    missing = [c for c in ["Precision", "Recall", "Test images"] if c not in shown.columns]
    if missing:
        st.caption("This metrics file does not include: " + ", ".join(missing) + ".")
    if "Test images" in shown.columns:
        small = [c for c, n in shown["Test images"].items() if pd.notna(n) and n < 30]
        if small:
            st.warning("Few images for: **" + ", ".join(small) + "**. "
                       "Their scores are noisy: one wrong prediction changes them a lot.")
else:
    waiting("The per-class table", "It needs per-class scores in the metrics file.")

st.subheader("3. Confusion Matrix")
left, right = st.columns([3, 2])
with left:
    cm_path = find_figure(f"{MODEL_NAME}_test_confusion*.png", "*test_confusion*.png")
    if cm_path:
        st.image(str(cm_path))
        st.caption("Test set, row-normalized: each row adds up to 1. The diagonal "
                   "is the share of each true class that was predicted correctly.")
    else:
        waiting("The confusion matrix", "Save it to `docs/figures/` as "
                f"`{MODEL_NAME}_test_confusion.png`.")
with right:
    st.markdown("**What to look for**")
    st.markdown(
        "- The diagonal is correct predictions; values off the diagonal are mistakes.\n"
        "- Look at which classes get mixed up, especially **crack vs damaged "
        "asphalt** and **pothole vs damaged asphalt**.\n"
        "- A low diagonal value means the model often misses that class."
    )

st.subheader("4. Model Comparison (Validation Scores)")
comp = comparison_table()
if comp is not None:
    st.dataframe(comp, column_config={
        "Accuracy": st.column_config.ProgressColumn(
            "Accuracy", min_value=0.0, max_value=1.0, format="%.2f"),
        "Macro F1": st.column_config.ProgressColumn(
            "Macro F1", min_value=0.0, max_value=1.0, format="%.2f"),
    })
    st.bar_chart(comp)
    st.caption("Scores are read from each model's `metrics_*.json` file in "
               "`docs/data_reports/`.")
    best = comp["Macro F1"].idxmax()
    if len(comp) > 1:
        if best == chosen_label:
            st.success(f"Chosen model: **{chosen_label}**. It has the highest "
                       "macro F1 of the models compared.")
        else:
            st.info(f"Chosen model: **{chosen_label}**. Note: **{best}** has a "
                    "higher macro F1 in these files.")
else:
    waiting("The model comparison", "It needs `metrics_*.json` files with "
            "accuracy and macro F1 in `docs/data_reports/`.")

st.subheader("5. Learning Curves (Chosen Model)")
curves = find_figure(f"{MODEL_NAME}*curve*.png", f"{MODEL_NAME}*history*.png",
                     "*learning_curve*.png", "*training_curve*.png")
if curves:
    st.image(str(curves))
    st.caption("Train vs validation accuracy and loss. The dashed line marks "
               "where fine-tuning starts. A growing gap between train and "
               "validation means overfitting.")
else:
    waiting("The learning curves", "Save the accuracy/loss figure to "
            f"`docs/figures/` as `{MODEL_NAME}_curves.png`.")
setup = setup_text(main["setup"])
if setup:
    st.caption("Training setup: " + setup)

st.subheader("6. Grad-CAM Examples")


def show_cams(paths):
    cols = st.columns(3)
    for i, path in enumerate(paths):
        label = path.stem.replace("gradcam_", "", 1).replace("_", " ")
        with cols[i % 3]:
            st.image(str(path), caption=label)


cams = sorted(FIG_DIR.glob("gradcam*.png"))
if cams:
    correct = [p for p in cams if p.stem.endswith("_correct")]
    wrong = [p for p in cams if "_predicted_as_" in p.stem]
    other = [p for p in cams if p not in correct and p not in wrong]
    if correct:
        st.markdown("**Correct predictions**")
        show_cams(correct)
    if wrong:
        st.markdown("**Wrong predictions**")
        show_cams(wrong)
    if other:
        st.markdown("**Other examples**")
        show_cams(other)
    st.caption("The highlighted areas show where the model looked when it made "
               "its decision. Wrong predictions help explain which classes "
               "get confused.")
else:
    waiting("Grad-CAM examples", "Save images named `gradcam_*.png` in "
            "`docs/figures/`. Include some wrong predictions.")

st.subheader("7. Limitations")
st.info(
    "- The **normal road** class was generated from unlabeled areas of the "
    "dataset images, so it may not represent all normal roads.\n"
    "- Classes with few images have less reliable scores.\n"
    "- Results come from one dataset and may not hold for other cities, "
    "cameras, or lighting."
)