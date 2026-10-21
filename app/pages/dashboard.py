import streamlit as st

st.set_page_config(page_title="Road Surface Dashboard", layout="wide")

CLASSES = [
    "crack",
    "pothole",
    "damaged asphalt",
    "water-filled pothole",
    "open manhole",
    "normal",
]


def placeholder(title, height=250, note="No data yet"):
    """A blank card where a chart will go later."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(
            f"<div style='height:{height}px;display:flex;align-items:center;"
            f"justify-content:center;opacity:.5;border:1px dashed gray;"
            f"border-radius:8px'>{note}</div>",
            unsafe_allow_html=True,
        )


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Filters")
    selected = st.multiselect("Classes", CLASSES, default=CLASSES)
    st.selectbox("Model", ["Baseline CNN", "MobileNetV2"], disabled=True)
    st.caption("Filters will work once the backend is connected.")

# ---------- Header ----------
st.title("Road Surface Condition Dashboard")
st.caption("Overview of model predictions per road-surface class")

# ---------- KPI row ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Images", "—")
k2.metric("Accuracy", "—")
k3.metric("Macro F1", "—")
k4.metric("Classes", len(CLASSES))

# ---------- Overall charts ----------
st.subheader("Overall")
c1, c2 = st.columns(2)
with c1:
    placeholder("Class Distribution (bar chart)")
with c2:
    placeholder("Confusion Matrix (heatmap)")

c3, c4 = st.columns(2)
with c3:
    placeholder("Precision / Recall / F1 per Class (grouped bar)")
with c4:
    placeholder("Training vs Validation Curve (line chart)")

# ---------- Per-class section ----------
st.subheader("Per Class")
if selected:
    tabs = st.tabs([c.title() for c in selected])
    for tab, cls in zip(tabs, selected):
        with tab:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Images", "—")
            m2.metric("Precision", "—")
            m3.metric("Recall", "—")
            m4.metric("F1-score", "—")

            a, b = st.columns(2)
            with a:
                placeholder(f"{cls.title()}: Confidence Distribution (histogram)")
            with b:
                placeholder(f"{cls.title()}: Most Confused With (bar chart)")

            placeholder(f"{cls.title()}: Sample Images / Grad-CAM", height=180)
else:
    st.info("Select at least one class in the sidebar.")