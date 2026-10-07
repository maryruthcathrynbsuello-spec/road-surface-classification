import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Performance", layout="wide")

CLASSES = [
    "crack",
    "pothole",
    "damaged asphalt",
    "water-filled pothole",
    "open manhole",
    "normal",
]


# ---------------------------------------------------------------
# MOCK DATA (fake numbers, replace with real results later)
# ---------------------------------------------------------------
@st.cache_data
def make_mock_data(seed=42):
    rng = np.random.default_rng(seed)
    n = len(CLASSES)

    
    counts = np.array([60, 70, 50, 40, 11, 90])
    cm = np.zeros((n, n), dtype=int)
    for i, c in enumerate(counts):
        correct = int(round(c * rng.uniform(0.85, 0.95)))
        cm[i, i] = correct
        weights = rng.random(n)
        weights[i] = 0
        weights /= weights.sum()
        cm[i] += rng.multinomial(c - correct, weights)
    
    for a, b, k in [(0, 2, 4), (2, 0, 3), (1, 3, 4), (3, 1, 3)]:
        k = min(k, cm[a, a] - 1)
        cm[a, b] += k
        cm[a, a] -= k

    tp = np.diag(cm)
    precision = tp / cm.sum(axis=0)
    recall = tp / cm.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    table = pd.DataFrame({
        "Test images": counts,
        "Precision": precision.round(3),
        "Recall": recall.round(3),
        "F1": f1.round(3),
    }, index=CLASSES)
    table.index.name = "Class"

    
    comparison = pd.DataFrame({
        "Val accuracy": [0.78, 0.89, 0.91],
        "Val macro F1": [0.74, 0.86, 0.88],
    }, index=["Baseline CNN", "MobileNetV2", "EfficientNetB0"])

    
    ep = np.arange(1, 21)
    acc = pd.DataFrame({
        "Train": 1 - 0.65 * np.exp(-ep / 5) + rng.normal(0, 0.004, 20),
        "Validation": 1 - 0.70 * np.exp(-ep / 6) - 0.05 + rng.normal(0, 0.008, 20),
    }, index=ep)
    loss = pd.DataFrame({
        "Train": 1.4 * np.exp(-ep / 5) + 0.05 + rng.normal(0, 0.01, 20),
        "Validation": 1.4 * np.exp(-ep / 6) + 0.20 + rng.normal(0, 0.02, 20),
    }, index=ep)
    acc.index.name = loss.index.name = "Epoch"

    return {
        "cm": cm,
        "table": table,
        "comparison": comparison,
        "acc": acc,
        "loss": loss,
        "test_acc": tp.sum() / cm.sum(),
        "macro_f1": f1.mean(),
    }


def placeholder(title, height=200, note="No data yet"):
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



with st.sidebar:
    use_mock = st.toggle("Use mock data", value=True)

data = make_mock_data() if use_mock else None


st.title("Model Performance")
st.caption("How well does the model classify unseen road-surface images?"
           + ("  |  MOCK DATA" if use_mock else ""))


st.subheader("1. Headline Numbers (Test Set)")
h1, h2, h3 = st.columns(3)
h1.metric("Test Accuracy", f"{data['test_acc']:.1%}" if data else "—")
h2.metric("Test Macro F1", f"{data['macro_f1']:.2f}" if data else "—")
h3.metric("Test Images", int(data["table"]["Test images"].sum()) if data else "—")
st.caption("These numbers come from the test set, images the model never saw "
           "during training or tuning.")


st.subheader("2. Per-Class Results")
if data:
    st.dataframe(data["table"])
    st.warning("Open manhole has very few test images, so its scores are noisy. "
               "One wrong prediction changes them a lot.")
else:
    placeholder("Per-class table (precision, recall, F1, test images)")


st.subheader("3. Confusion Matrix")
left, right = st.columns([3, 2])
with left:
    if data:
        with st.container(border=True):
            st.pyplot(confusion_figure(data["cm"]))
    else:
        placeholder("Confusion matrix (heatmap)", height=300)
with right:
    st.markdown("**What to look for**")
    st.markdown(
        "- **Crack vs damaged asphalt:** similar cracked textures\n"
        "- **Pothole vs water-filled pothole:** the water changes the look\n"
        "- Numbers outside the diagonal are mistakes."
    )

st.subheader("4. Model Comparison (Validation Scores)")
if data:
    st.bar_chart(data["comparison"])
    best = data["comparison"]["Val macro F1"].idxmax()
    st.success(f"Chosen model: **{best}** (highest validation macro F1). "
               "The test set is used only once, for the chosen model.")
else:
    placeholder("Baseline CNN vs MobileNetV2 vs EfficientNetB0 (bar chart)")


st.subheader("5. Learning Curves (Chosen Model)")
c1, c2 = st.columns(2)
with c1:
    if data:
        with st.container(border=True):
            st.markdown("**Accuracy: Train vs Validation**")
            st.line_chart(data["acc"])
    else:
        placeholder("Accuracy curve (line chart)")
with c2:
    if data:
        with st.container(border=True):
            st.markdown("**Loss: Train vs Validation**")
            st.line_chart(data["loss"])
    else:
        placeholder("Loss curve (line chart)")
if data:
    gap = data["acc"]["Train"].iloc[-1] - data["acc"]["Validation"].iloc[-1]
    st.caption(f"Final gap between train and validation accuracy: {gap:.1%}. "
               "A small gap means little overfitting.")


st.subheader("6. Grad-CAM Examples")
g1, g2, g3 = st.columns(3)
with g1:
    placeholder("Correct prediction", height=160, note="Image + heatmap")
with g2:
    placeholder("Correct prediction", height=160, note="Image + heatmap")
with g3:
    placeholder("Wrong prediction", height=160, note="Image + heatmap")
st.caption("Needs the trained model. Include some wrong predictions to show "
           "what the model got confused by.")


st.subheader("7. Limitations")
st.info(
    "- The **normal road** class was generated from unlabeled areas of the "
    "dataset images, so it may not represent all normal roads.\n"
    "- **Open manhole** has very few examples, so its results are less reliable.\n"
    "- Results come from one dataset and may not hold for other cities, "
    "cameras, or lighting."
)