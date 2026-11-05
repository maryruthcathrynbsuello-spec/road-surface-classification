import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Performance", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures"
REPORT_DIR = ROOT / "docs" / "data_reports"

MODEL_NAME = "efficientnetb0_v2"
METRICS_FILE = REPORT_DIR / f"{MODEL_NAME}_metrics.json"


def find_figure(*patterns):
    """Return the first figure in docs/figures matching any pattern, else None."""
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


def pretty(name):
    return name.replace("_", " ")


def waiting(what, hint):
    st.info(f"**{what}** is not available yet. {hint}")


metrics = load_metrics()

st.title("Model Performance")
st.caption(f"Results of the chosen model ({MODEL_NAME}) on unseen test images.")

if metrics is None:
    st.warning(
        f"Metrics file not found: `docs/data_reports/{METRICS_FILE.name}`. "
        "Numbers and tables will appear once it is added. "
        "Figures are loaded separately from `docs/figures/`."
    )

st.subheader("1. Headline Numbers (Test Set)")
h1, h2, h3 = st.columns(3)
if metrics:
    per_class = metrics.get("per_class", {})
    total = sum(v.get("support", 0) for v in per_class.values())
    h1.metric("Test Accuracy", f"{metrics['test_accuracy']:.1%}")
    h2.metric("Test Macro F1", f"{metrics['test_macro_f1']:.2f}")
    h3.metric("Test Images", f"{total:,}")
else:
    h1.metric("Test Accuracy", "—")
    h2.metric("Test Macro F1", "—")
    h3.metric("Test Images", "—")
st.caption("These numbers come from the test set: images the model never saw "
           "during training or tuning.")

st.subheader("2. Per-Class Results")
if metrics and metrics.get("per_class"):
    df = pd.DataFrame(metrics["per_class"]).T
    df = df.rename(columns={"support": "Test images", "precision": "Precision",
                            "recall": "Recall", "f1": "F1"})
    df = df[["Test images", "Precision", "Recall", "F1"]]
    df.index = [pretty(c) for c in df.index]
    df.index.name = "Class"
    st.dataframe(df.round(3))

    small = [pretty(c) for c, v in metrics["per_class"].items()
             if v.get("support", 0) < 30]
    if small:
        st.warning("Few test images for: **" + ", ".join(small) + "**. "
                   "Their scores are noisy: one wrong prediction changes them a lot.")
else:
    waiting("The per-class table", "It needs precision, recall, F1 and the "
            "number of test images per class in the metrics file.")

st.subheader("3. Confusion Matrix")
left, right = st.columns([3, 2])
with left:
    cm_path = find_figure(f"{MODEL_NAME}_test_confusion*.png", "*test_confusion*.png")
    if cm_path:
        st.image(str(cm_path))
        st.caption("Row-normalized: each row adds up to 1. The diagonal is the "
                   "share of each true class that was predicted correctly.")
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
comparison = metrics.get("comparison") if metrics else None
if comparison:
    cdf = pd.DataFrame(comparison).T.rename(columns={
        "val_accuracy": "Val accuracy", "val_macro_f1": "Val macro F1"})
    st.bar_chart(cdf)
    chosen = metrics.get("chosen_model")
    reason = metrics.get("chosen_reason", "")
    if chosen:
        st.success(f"Chosen model: **{chosen}**. {reason}")
else:
    waiting("The model comparison", "Add a `comparison` entry (Baseline CNN, "
            "MobileNetV2, EfficientNetB0 validation scores) to the metrics file.")

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

st.subheader("6. Grad-CAM Examples")
cams = sorted(FIG_DIR.glob("gradcam*.png"))
if cams:
    cols = st.columns(min(3, len(cams)))
    for i, path in enumerate(cams[:6]):
        with cols[i % len(cols)]:
            st.image(str(path), caption=path.stem.replace("_", " "))
else:
    waiting("Grad-CAM examples", "Save images named `gradcam_*.png` in "
            "`docs/figures/`. Include some wrong predictions.")

st.subheader("7. YOLO Detection")
yolo_figs = sorted(FIG_DIR.glob("yolo*.png"))
yolo_metrics_file = REPORT_DIR / "yolo_metrics.json"
if yolo_figs or yolo_metrics_file.exists():
    if yolo_metrics_file.exists():
        try:
            ym = json.loads(yolo_metrics_file.read_text(encoding="utf-8"))
            if "validation" in ym:
                st.markdown("**Validation**")
                st.dataframe(pd.DataFrame(ym["validation"]).T.round(3))
            if "test_map" in ym:
                st.metric("YOLO test mAP", f"{ym['test_map']:.3f}")
        except json.JSONDecodeError as e:
            st.error(f"Could not read yolo_metrics.json: {e}")
    cols = st.columns(2)
    for i, path in enumerate(yolo_figs):
        with cols[i % 2]:
            st.image(str(path), caption=path.stem.replace("_", " "))
else:
    st.info("YOLO results are not available yet. Training curves, PR curve, "
            "confusion matrix and the validation table will appear here once "
            "`yolo_*.png` figures and `yolo_metrics.json` are added. "
            "Test mAP is shown only after the test run.")

st.subheader("8. Limitations")
st.info(
    "- The **normal road** class was generated from unlabeled areas of the "
    "dataset images, so it may not represent all normal roads.\n"
    "- Classes with few test images have less reliable scores.\n"
    "- Results come from one dataset and may not hold for other cities, "
    "cameras, or lighting."
)