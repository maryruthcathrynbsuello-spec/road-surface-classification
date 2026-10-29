import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Road Surface Dashboard", layout="wide")

CLASSES = [
    "crack",
    "pothole",
    "damaged asphalt",
    "water-filled pothole",
    "open manhole",
    "normal",
]


# ---------------------------------------------------------------
# MOCK DATA (fake numbers, replace with real backend output later)
# ---------------------------------------------------------------
@st.cache_data
def make_mock_data(seed=42):
    rng = np.random.default_rng(seed)
    n = len(CLASSES)

    # Test images per class
    counts = np.array([180, 220, 150, 120, 90, 260])

    # Confusion matrix: mostly correct, with the confusions from the project plan
    cm = np.zeros((n, n), dtype=int)
    for i, c in enumerate(counts):
        correct = int(c * rng.uniform(0.82, 0.94))
        cm[i, i] = correct
        rest = c - correct
        weights = rng.random(n)
        weights[i] = 0
        weights /= weights.sum()
        cm[i] += rng.multinomial(rest, weights)
    cm[0, 2] += 10; cm[0, 0] -= 10      # crack -> damaged asphalt
    cm[2, 0] += 8;  cm[2, 2] -= 8       # damaged asphalt -> crack
    cm[1, 3] += 9;  cm[1, 1] -= 9       # pothole -> water-filled pothole
    cm[3, 1] += 7;  cm[3, 3] -= 7       # water-filled -> pothole

    tp = np.diag(cm)
    precision = tp / cm.sum(axis=0)
    recall = tp / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    metrics = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1": f1}, index=CLASSES
    )

    # Training curves (20 epochs)
    epochs = np.arange(1, 21)
    train_acc = 1 - 0.65 * np.exp(-epochs / 5) + rng.normal(0, 0.005, 20)
    val_acc = 1 - 0.70 * np.exp(-epochs / 6) - 0.04 + rng.normal(0, 0.01, 20)
    curves = pd.DataFrame(
        {"Train accuracy": train_acc, "Validation accuracy": val_acc},
        index=epochs,
    )
    curves.index.name = "Epoch"

    # Confidence scores per class
    conf = {
        c: np.clip(rng.beta(a, 1.5, size=counts[i]), 0, 1)
        for i, (c, a) in enumerate(zip(CLASSES, [6, 7, 5, 5, 8, 9]))
    }

    return {
        "counts": pd.Series(counts, index=CLASSES),
        "cm": cm,
        "metrics": metrics,
        "curves": curves,
        "conf": conf,
        "accuracy": tp.sum() / cm.sum(),
        "macro_f1": f1.mean(),
    }


def placeholder(title, height=250, note="No data yet"):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(
            f"<div style='height:{height}px;display:flex;align-items:center;"
            f"justify-content:center;opacity:.5;border:1px dashed gray;"
            f"border-radius:8px'>{note}</div>",
            unsafe_allow_html=True,
        )


def confusion_figure(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Filters")
    use_mock = st.toggle("Use mock data", value=True)
    selected = st.multiselect("Classes", CLASSES, default=CLASSES)
    st.selectbox("Model", ["Baseline CNN", "MobileNetV2"], disabled=True)

data = make_mock_data() if use_mock else None

# ---------- Header ----------
st.title("Road Surface Condition Dashboard")
st.caption("Overview of model predictions per road-surface class"
           + ("  |  MOCK DATA" if use_mock else ""))

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Images", f"{data['counts'].sum():,}" if data else "—")
k2.metric("Accuracy", f"{data['accuracy']:.1%}" if data else "—")
k3.metric("Macro F1", f"{data['macro_f1']:.2f}" if data else "—")
k4.metric("Classes", len(CLASSES))

# ---------- Overall ----------
st.subheader("Overall")
c1, c2 = st.columns(2)
with c1:
    if data:
        with st.container(border=True):
            st.markdown("**Class Distribution**")
            st.bar_chart(data["counts"])
    else:
        placeholder("Class Distribution (bar chart)")
with c2:
    if data:
        with st.container(border=True):
            st.markdown("**Confusion Matrix**")
            st.pyplot(confusion_figure(data["cm"]))
    else:
        placeholder("Confusion Matrix (heatmap)")

c3, c4 = st.columns(2)
with c3:
    if data:
        with st.container(border=True):
            st.markdown("**Precision / Recall / F1 per Class**")
            st.bar_chart(data["metrics"])
    else:
        placeholder("Precision / Recall / F1 per Class (grouped bar)")
with c4:
    if data:
        with st.container(border=True):
            st.markdown("**Training vs Validation Accuracy**")
            st.line_chart(data["curves"])
    else:
        placeholder("Training vs Validation Curve (line chart)")

# ---------- Per class ----------
st.subheader("Per Class")
if selected:
    tabs = st.tabs([c.title() for c in selected])
    for tab, cls in zip(tabs, selected):
        with tab:
            i = CLASSES.index(cls)
            m1, m2, m3, m4 = st.columns(4)
            if data:
                m = data["metrics"].loc[cls]
                m1.metric("Images", int(data["counts"][cls]))
                m2.metric("Precision", f"{m['Precision']:.2f}")
                m3.metric("Recall", f"{m['Recall']:.2f}")
                m4.metric("F1-score", f"{m['F1']:.2f}")
            else:
                for col, label in zip((m1, m2, m3, m4),
                                      ("Images", "Precision", "Recall", "F1-score")):
                    col.metric(label, "—")

            a, b = st.columns(2)
            with a:
                if data:
                    with st.container(border=True):
                        st.markdown(f"**{cls.title()}: Confidence Distribution**")
                        hist, edges = np.histogram(
                            data["conf"][cls], bins=10, range=(0, 1))
                        labels = [f"{edges[k]:.1f}-{edges[k+1]:.1f}"
                                  for k in range(10)]
                        st.bar_chart(pd.Series(hist, index=labels))
                else:
                    placeholder(f"{cls.title()}: Confidence Distribution (histogram)")
            with b:
                if data:
                    with st.container(border=True):
                        st.markdown(f"**{cls.title()}: Most Confused With**")
                        row = pd.Series(data["cm"][i], index=CLASSES).drop(cls)
                        st.bar_chart(row)
                else:
                    placeholder(f"{cls.title()}: Most Confused With (bar chart)")

            placeholder(f"{cls.title()}: Sample Images / Grad-CAM", height=180,
                        note="Needs real images from the backend")
else:
    st.info("Select at least one class in the sidebar.")