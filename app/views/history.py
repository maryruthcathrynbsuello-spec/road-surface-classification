import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parents[1]))  # app/ folder, so `utils` can be imported
from utils.inference import pretty

HISTORY_FILE = Path(__file__).with_name("history.json")
IMAGES_DIR = HISTORY_FILE.with_name("history_images")

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


def delete_images(records):
    for record in records:
        if record.get("image"):
            (IMAGES_DIR / record["image"]).unlink(missing_ok=True)


@st.dialog("Analysis details", width="large")
def show_details(record):
    left, right = st.columns(2)

    with left:
        image_path = IMAGES_DIR / record["image"] if record.get("image") else None
        if image_path and image_path.exists():
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            st.image(image, use_container_width=True)
        else:
            st.info("The image for this analysis isn't available.")

    with right:
        st.caption(record["file_name"])
        recognized = record.get("recognized", record["status"] == "OK")
        if recognized:
            st.subheader(pretty(record["labels"][0]).upper())
        else:
            st.subheader("NOT RECOGNIZED")

        confidence = record.get("confidence")
        if confidence is not None:
            st.write(f"Confidence: **{confidence:.1f}%**")
            st.progress(min(max(confidence / 100, 0.0), 1.0))

        if recognized:
            if record.get("action"):
                st.caption(record["action"])
        elif record.get("best_guess") is not None and record.get("threshold") is not None:
            st.warning(
                f"Not sure (best guess: {pretty(record['best_guess'])}, below the "
                f"{record['threshold']:.0f}% threshold). Try a clearer, well-lit close-up."
            )
        else:
            st.warning("Not sure about this result. Try a clearer, well-lit close-up.")

        if record.get("top3"):
            st.write("**Top 3**")
            for name, pct in record["top3"]:
                st.write(f"{pretty(name)}: {pct:.1f}%")

        st.caption(f"Analyzed on {record['time']}")


# Compact table rows + file names that look like links (underline on hover)
st.markdown(
    """
    <style>
    .st-key-history_table {
        gap: 0.15rem;
        max-height: min(60vh, 600px);
        overflow-y: auto;
        overflow-x: hidden;
        overscroll-behavior: contain;
    }
    .st-key-history_table [data-testid="stHorizontalBlock"] {
        gap: 0.7rem;
        padding: 0.35rem 0.4rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    }
    .st-key-history_table [data-testid="stHorizontalBlock"] > [data-testid="column"] + [data-testid="column"] {
        border-left: 1px solid rgba(128, 128, 128, 0.18);
        padding-left: 0.45rem;
    }
    .st-key-history_table [data-testid="stCheckbox"] { min-height: 0; }
    .st-key-history_table p { margin: 0; }
    .st-key-history_table .history-table-header { transform: translateY(-8px); }
    .st-key-history_table [data-testid="stBaseButton-tertiary"] {
        padding: 0;
        min-height: 0;
        height: auto;
        justify-content: flex-start;
        background: transparent;
    }
    .st-key-history_table [data-testid="stBaseButton-tertiary"]:hover p {
        text-decoration: underline;
    }
    div[data-testid="stDialog"] div[role="dialog"] {
        max-height: 90vh;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("History")
st.write("Every analysis from the Upload tab is saved here. Click a file name to see its details.")

records = list(reversed(load_history()))

if not records:
    st.info("No history yet. Classify some images on the Upload tab first.")
    st.stop()

all_labels = sorted({label for record in records for label in record["labels"]})

STATUS_OPTIONS = {
    "All": None,
    "Needs review": "Needs review",
    "OK": "OK",
}

f1, f2 = st.columns(2)
chosen_labels = f1.multiselect("Filter by class", all_labels)
chosen_status_label = f2.selectbox("Filter by review", list(STATUS_OPTIONS))
chosen_status = STATUS_OPTIONS[chosen_status_label]

filtered = [
    record
    for record in records
    if (not chosen_labels or any(label in chosen_labels for label in record["labels"]))
    and (chosen_status is None or record["status"] == chosen_status)
]

st.caption(f"Showing {len(filtered)} of {len(records)} analyses")

if not filtered:
    st.warning("Nothing matches those filters.")
    st.stop()

# ---------- Table ----------
WIDTHS = [0.45, 3.2, 1.8, 1.1, 2.1, 1.35]

selected_ids = set()
opened = None

with st.container(key="history_table"):
    header = st.columns(WIDTHS, vertical_alignment="center")
    for col, title in zip(header[1:], ["File", "Time", "Confidence", "Labels", "Status"]):
        col.markdown(
            f'<div class="history-table-header"><strong>{title}</strong></div>',
            unsafe_allow_html=True,
        )

    for record in filtered:
        cols = st.columns(WIDTHS, vertical_alignment="center")

        if cols[0].checkbox("Select", key=f"sel_{record['id']}", label_visibility="collapsed"):
            selected_ids.add(record["id"])

        if cols[1].button(record["file_name"], key=f"open_{record['id']}", type="tertiary"):
            opened = record

        cols[2].write(record["time"])
        confidence = record.get("confidence")
        cols[3].write(f"{confidence:.1f}%" if confidence is not None else "")
        cols[4].write(", ".join(record["labels"]))
        cols[5].write(record["status"])

if opened is not None:
    show_details(opened)

# ---------- Actions ----------
d1, d2, _ = st.columns([1, 1, 3])

if d1.button(
    f"Delete selected ({len(selected_ids)})",
    disabled=not selected_ids,
    use_container_width=True,
):
    everything = load_history()
    delete_images([r for r in everything if r["id"] in selected_ids])
    save_history([r for r in everything if r["id"] not in selected_ids])
    st.rerun()

if d2.button("Clear all history", use_container_width=True):
    delete_images(load_history())
    save_history([])
    st.rerun()