import json
from pathlib import Path

import pandas as pd
import streamlit as st

HISTORY_FILE = Path(__file__).with_name("history.json")

st.set_page_config(layout="wide")


def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        records = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read {HISTORY_FILE.name}: {error}")
        st.stop()
    if not isinstance(records, list):
        st.error(f"Expected {HISTORY_FILE.name} to contain a list of records.")
        st.stop()
    return records


def save_history(records):
    HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


st.title("History")
st.write("Every analysis from the Upload tab is saved here.")

records = list(reversed(load_history()))

if not records:
    st.info("No history yet. Classify some images on the Upload tab first.")
    st.stop()

all_labels = sorted({label for record in records for label in record["labels"]})
all_status = sorted({record["status"] for record in records})

f1, f2 = st.columns(2)
chosen_labels = f1.multiselect("Filter by class", all_labels)
chosen_status = f2.multiselect("Filter by review", all_status)

filtered = [
    record
    for record in records
    if (not chosen_labels or any(label in chosen_labels for label in record["labels"]))
    and (not chosen_status or record["status"] in chosen_status)
]

st.caption(f"Showing {len(filtered)} of {len(records)} analyses")

if not filtered:
    st.warning("Nothing matches those filters.")
    st.stop()

df = pd.DataFrame(
    {
        "Time": [record["time"] for record in filtered],
        "File": [record["file_name"] for record in filtered],
        "Confidence": [
            f"{record['confidence']:.1f}%" if record.get("confidence") is not None else ""
            for record in filtered
        ],
        "Labels": [", ".join(record["labels"]) for record in filtered],
        "Status": [record["status"] for record in filtered],
    }
)

event = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
)

selected_ids = {filtered[index]["id"] for index in event.selection.rows}

d1, d2, _ = st.columns([1, 1, 3])

if d1.button(
    f"Delete selected ({len(selected_ids)})",
    disabled=not selected_ids,
    use_container_width=True,
):
    save_history([record for record in load_history() if record["id"] not in selected_ids])
    st.rerun()

if d2.button("Clear all history", use_container_width=True):
    save_history([])
    st.rerun()
