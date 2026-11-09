import io
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parents[1]))   # app/ folder, so `utils` can be imported
from utils.inference import (DEFAULT_THRESHOLD, MAX_FILE_MB, load_model, missing_files,
                             predict, pretty, validate_upload)

HISTORY_FILE = Path(__file__).with_name("history.json")  # same file history.py reads

# ---------- Page setup ----------
st.set_page_config(
    layout="wide",
)

ACTIONS = {
    "pothole": "Consider reporting this to your local roads authority.",
    "water-filled_pothole": "The depth is hidden by the water. Treat it like a pothole and consider reporting it.",
    "open_manhole": "An open manhole is a safety hazard. Report it promptly.",
    "crack": "Cracks can be monitored. Report them if they are wide or spreading.",
    "damaged_asphalt": "Consider reporting damaged pavement to your local roads authority.",
    "normal": "No defect recognized in this photo.",
}
DEFAULT_ACTION = "No specific action suggested for this result."

# ---------- State ----------
st.session_state.setdefault("images", [])     # classified images: [{"name", "bytes"}]
st.session_state.setdefault("results", None)  # list of predict() result dicts, or None
st.session_state.setdefault("uploader_key", 0)


def save_history(images, results):
    """Append one history row per image to history.json (survives refresh)."""
    try:
        history = json.loads(HISTORY_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for item, r in zip(images, results):
        history.append(
            {
                "id": uuid.uuid4().hex,
                "time": now,
                "file_name": item["name"],
                "confidence": round(r["confidence"], 1),
                "labels": [r["label"]],
                "status": "OK" if r["recognized"] else "Needs review",
            }
        )
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def reset():
    st.session_state.images = []
    st.session_state.results = None
    st.session_state.uploader_key += 1  # gives a fresh, empty uploader


def load_image(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def show_grid(images):
    cols = st.columns(3)
    for i, item in enumerate(images):
        with cols[i % 3]:
            st.image(load_image(item["bytes"]), use_container_width=True)
            st.caption(item["name"])


def dropzone_css(has_files: bool):
    """Make the drop zone big and roomy (compact once images are added)."""
    padding = "1.25rem 1rem" if has_files else "3rem 1rem"
    gap = "0.75rem" if has_files else "1.5rem"
    st.markdown(
        f"""
        <style>
        [data-testid="stFileUploaderDropzone"] {{
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            gap: {gap};
            padding: {padding};
            border: 2px dashed rgba(128, 128, 128, 0.6);
            border-radius: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("Classifies the condition of road surfaces from photos.")
    st.caption(f"Supported formats: JPG, JPEG, PNG (max {MAX_FILE_MB} MB each)")
    st.caption("Results are estimates and should not replace a professional inspection.")
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.10, 0.95, DEFAULT_THRESHOLD, 0.05,
                          help="Below this confidence the app answers 'Not recognized' instead of a label.")

# ---------- Model files must exist ----------
missing = missing_files()
if missing:
    st.error("Model files not found. Copy them to app/models/ and reload:\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()

# ---------- Header ----------
st.title("Road Surface Classifier")
st.write("Upload one or more images of road surfaces to classify their condition.")
st.caption("Works best on a close-up of one road area. A photo with several different defects gets one label.")

# ---------- Main layout (same two columns at every stage) ----------
left, right = st.columns([3, 2], border=True)

# ----- Left column: uploader (stays available) + image grid -----
with left:
    header = st.empty()

    if st.session_state.results is None:
        uploaded_files = st.file_uploader(
            "Drag and drop images here, or browse files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        images, errors = [], []
        for f in uploaded_files or []:
            data = f.getvalue()
            _, error = validate_upload(f.name, data)   # type, size, corrupted or non-image checks
            if error:
                errors.append(error)
            else:
                images.append({"name": f.name, "bytes": data})

        dropzone_css(has_files=bool(images))
        for error in errors:
            st.error(error)
        show_grid(images)
    else:
        images = st.session_state.images
        show_grid(images)

    count = len(images)
    header.markdown("**Uploaded Images**" + (f" ({count})" if count else ""))

# Determine current stage
if st.session_state.results is not None:
    stage = 3
elif not images:
    stage = 1
else:
    stage = 2

# ----- Right column: status -> action -> results -----
with right:
    st.markdown("**Classification**")

    if stage == 1:
        st.info("Upload images to begin.")

    elif stage == 2:
        n = len(images)
        st.write(f"Ready to classify {n} image{'s' if n != 1 else ''}")
        st.caption("You can keep adding more images on the left.")

        if st.button("Classify Images", use_container_width=True):
            with st.spinner("Loading the model (first run only)..."):
                bundle = load_model()                        # cached: loaded once, reused afterwards
            results = []
            progress = st.progress(0.0, text="Analyzing road surfaces...")
            for i, item in enumerate(images, start=1):
                results.append(predict(load_image(item["bytes"]), threshold=threshold, bundle=bundle))
                progress.progress(i / n, text=f"Analyzing {i} of {n}...")

            # Save to persistent history (shown on the History tab)
            save_history(images, results)

            st.session_state.images = images
            st.session_state.results = results
            st.rerun()

        st.button("Clear all images", on_click=reset, type="tertiary")

    else:
        st.caption("Classification Results")

        for item, r in zip(st.session_state.images, st.session_state.results):
            with st.container(border=True):
                st.caption(item["name"])
                if r["recognized"]:
                    st.subheader(pretty(r["label"]).upper())
                else:
                    st.subheader("NOT RECOGNIZED")
                st.write(f"Confidence: **{r['confidence']:.1f}%**")
                st.progress(min(max(r["confidence"] / 100, 0.0), 1.0))

                if r["recognized"]:
                    st.caption(ACTIONS.get(r["label"], DEFAULT_ACTION))
                else:
                    st.warning(f"Not sure (best guess: {pretty(r['best_guess'])}, below the "
                               f"{r['threshold']:.0f}% threshold). Try a clearer, well-lit close-up.")

                st.write("**Top 3**")
                for name, pct in r["top3"]:
                    st.write(f"{pretty(name)}: {pct:.1f}%")

        st.button("Upload More Images", on_click=reset, use_container_width=True)

# ---------- Footer ----------
with st.expander("Tips for a good photo"):
    st.write(
        "- Shoot straight down or at a slight angle.\n"
        "- Use daylight and avoid heavy shadows.\n"
        "- Let the road surface fill most of the frame.\n"
        "- Keep the camera steady to avoid blur."
    )