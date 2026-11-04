import json
from pathlib import Path

import pandas as pd
import streamlit as st

HISTORY_FILE = Path(__file__).with_name("history.json")  # written by upload.py

st.set_page_config(layout="wide")


def load_history():
    try:
        return json.loads(HISTORY_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(records):
    HISTORY_FILE.write_text(json.dumps(records, indent=2))


st.title("History")
st.write("Every analysis from the Upload tab is saved here.")

records = list(reversed(load_history()))  # newest first

if not records:
    st.info("No history yet. Classify some images on the Upload tab first.")
    st.stop()

# ---------- Filters ----------
all_labels = sorted({lbl for r in records for lbl in r["labels"]})
all_status = sorted({r["status"] for r in records})

f1, f2 = st.columns(2)
chosen_labels = f1.multiselect("Filter by class", all_labels)
chosen_status = f2.multiselect("Filter by review", all_status)

filtered = [
    r
    for r in records
    if (not chosen_labels or any(lbl in chosen_labels for lbl in r["labels"]))
    and (not chosen_status or r["status"] in chosen_status)
]

st.caption(f"Showing {len(filtered)} of {len(records)} analyses")

if not filtered:
    st.warning("Nothing matches those filters.")
    st.stop()

# ---------- Table (select rows to delete) ----------
df = pd.DataFrame(
    {
        "Time": [r["time"] for r in filtered],
        "File": [r["file_name"] for r in filtered],
        "Detections": [r["detections"] for r in filtered],
        "Labels": [", ".join(r["labels"]) for r in filtered],
        "Status": [r["status"] for r in filtered],
    }
)

event = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
)

selected_ids = {filtered[i]["id"] for i in event.selection.rows}

# ---------- Delete ----------
d1, d2, _ = st.columns([1, 1, 3])

if d1.button(
    f"Delete selected ({len(selected_ids)})",
    disabled=not selected_ids,
    use_container_width=True,
):
    save_history([r for r in load_history() if r["id"] not in selected_ids])
    st.rerun()

if d2.button("Clear all history", use_container_width=True):
    save_history([])
    st.rerun()