import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))  # app/ folder, so `utils` can be imported
from utils.inference import DEFAULT_THRESHOLD, MAX_FILE_MB

HISTORY_FILE = Path(__file__).with_name("history.json")      # same file the Upload and History tabs use
IMAGES_DIR = HISTORY_FILE.with_name("history_images")

st.set_page_config(layout="wide")

# Edit these two to match your actual model
MODEL_NAME = "Road surface classifier"
MODEL_CLASSES = ["pothole", "water-filled pothole", "open manhole", "crack", "damaged asphalt", "normal"]

# Plain keys so values survive switching tabs
st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)
st.session_state.setdefault("show_top3", True)


def read_history():
    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return records if isinstance(records, list) else []
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []


@st.dialog("Clear all history?")
def confirm_clear():
    st.write("This permanently deletes every saved analysis and its stored image.")
    c1, c2 = st.columns(2)
    if c1.button("Yes, clear it", type="primary", use_container_width=True):
        HISTORY_FILE.write_text("[]", encoding="utf-8")
        shutil.rmtree(IMAGES_DIR, ignore_errors=True)
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.rerun()


st.title("Settings")

# ---------- Prediction display ----------
st.subheader("Prediction")
st.session_state.threshold = st.slider(
    "Confidence threshold",
    0.10,
    0.95,
    st.session_state.threshold,
    0.05,
    help="Below this confidence the app answers 'Not recognized' instead of a label.",
)
st.session_state.show_top3 = st.toggle(
    "Show top 3 possible classes",
    value=st.session_state.show_top3,
    help="Show the three most likely classes under each result on the Upload tab.",
)
st.caption("These apply to the next images you classify on the Upload tab.")

st.divider()

# ---------- Upload guidance (read-only) ----------
st.subheader("Upload guidance")
st.write("**Supported formats:** JPG, JPEG, PNG")
st.write(f"**Maximum file size:** {MAX_FILE_MB} MB per image")
st.write("**Best results:** a close-up of one road area, in daylight, with the surface filling most of the frame.")

st.divider()

# ---------- History controls ----------
st.subheader("History")
history = read_history()
st.write(f"{len(history)} saved analyses.")

h1, h2, h3, _ = st.columns([1, 1, 1, 2])

if history:
    h1.download_button(
        "Export JSON",
        data=json.dumps(history, indent=2),
        file_name="history.json",
        mime="application/json",
        use_container_width=True,
    )
    csv = pd.DataFrame(
        {
            "time": [r.get("time") for r in history],
            "file_name": [r.get("file_name") for r in history],
            "confidence": [r.get("confidence") for r in history],
            "labels": [", ".join(r.get("labels", [])) for r in history],
            "status": [r.get("status") for r in history],
        }
    ).to_csv(index=False)
    h2.download_button(
        "Export CSV",
        data=csv,
        file_name="history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if h3.button("Clear history", use_container_width=True):
        confirm_clear()
else:
    st.caption("Nothing to export or clear yet.")

st.divider()

# ---------- Model details (read-only) ----------
st.subheader("Model")
st.write(f"**Model:** {MODEL_NAME}")
st.write(f"**Classes:** {', '.join(MODEL_CLASSES)}")
st.caption(
    "Limitations: each photo gets one label even if it shows several defects. "
    "Results are estimates and should not replace a professional inspection. "
    "See the Model Performance tab for accuracy details."
)