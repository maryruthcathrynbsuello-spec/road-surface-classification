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
IMAGES_DIR = HISTORY_FILE.with_name("history_images")    # saved copies of uploaded images

PANEL_HEIGHT = 440   # fixed height (px) of the scrollable image grid / results list
GRID_COLUMNS = 5     # thumbnails per row in the grid

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
st.session_state.setdefault("images", [])         # uploaded images: [{"name", "bytes"}]
st.session_state.setdefault("results", None)      # list of predict() result dicts, or None
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("adding", False)      # True while the user is adding more images
st.session_state.setdefault("upload_errors", [])  # validation errors from the last upload
st.session_state.setdefault("threshold", DEFAULT_THRESHOLD)
st.session_state.setdefault("show_top3", True)


def save_history(images, results):
    """Append one history row per image to history.json and keep a copy of each image."""
    try:
        history = json.loads(HISTORY_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    IMAGES_DIR.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for item, r in zip(images, results):
        record_id = uuid.uuid4().hex
        image_file = f"{record_id}{Path(item['name']).suffix.lower() or '.jpg'}"
        (IMAGES_DIR / image_file).write_bytes(item["bytes"])
        history.append(
            {
                "id": record_id,
                "time": now,
                "file_name": item["name"],
                "confidence": round(r["confidence"], 1),
                "labels": [r["label"]],
                "status": "OK" if r["recognized"] else "Needs review",
                # extra details for the History pop-up
                "image": image_file,
                "recognized": r["recognized"],
                "best_guess": r["best_guess"],
                "threshold": r["threshold"],
                "top3": [[name, round(pct, 1)] for name, pct in r["top3"]],
                "action": ACTIONS.get(r["label"], DEFAULT_ACTION) if r["recognized"] else None,
            }
        )
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ---------- Callbacks ----------
def on_upload():
    """Runs as soon as files are dropped: validate, store them, then move on to the next step."""
    files = st.session_state.get(f"uploader_{st.session_state.uploader_key}") or []
    errors = []
    for f in files:
        data = f.getvalue()
        _, error = validate_upload(f.name, data)   # type, size, corrupted or non-image checks
        if error:
            errors.append(error)
        else:
            st.session_state.images.append({"name": f.name, "bytes": data})
    st.session_state.upload_errors = errors
    st.session_state.results = None
    st.session_state.adding = False
    st.session_state.uploader_key += 1            # fresh, empty uploader for next time


def reset():
    st.session_state.images = []
    st.session_state.results = None
    st.session_state.adding = False
    st.session_state.upload_errors = []
    st.session_state.uploader_key += 1


def start_adding():
    st.session_state.adding = True
    st.session_state.upload_errors = []


def cancel_adding():
    st.session_state.adding = False


def remove_image(index: int):
    if 0 <= index < len(st.session_state.images):
        st.session_state.images.pop(index)
    if not st.session_state.images:
        reset()


def new_batch():
    """Back to the upload panel after viewing results."""
    reset()


# ---------- Helpers ----------
def load_image(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=200)
def make_thumbnail(data: bytes, size: int = 360) -> Image.Image:
    """Square, center-cropped thumbnail so every image has the same size (cached per image)."""
    return ImageOps.fit(load_image(data), (size, size), method=Image.Resampling.LANCZOS)


@st.cache_data(show_spinner=False, max_entries=20)
def make_preview(data: bytes, size: int = 1400) -> Image.Image:
    """Large version for the pop-up; keeps the original proportions (no cropping)."""
    img = load_image(data)
    img.thumbnail((size, size))
    return img


@st.dialog("Image preview", width="large")
def view_image(index: int):
    """Pop-up with a bigger view of one uploaded image."""
    images = st.session_state.images
    if not 0 <= index < len(images):
        st.info("This image is no longer available.")
        return
    item = images[index]
    st.caption(f"{item['name']}  ({index + 1} of {len(images)})")
    st.image(make_preview(item["bytes"]), use_container_width=True)


@st.dialog("Classification result", width="large")
def view_result(index: int):
    """Pop-up with the full image and the complete result for one image."""
    images, results = st.session_state.images, st.session_state.results
    if not results or not 0 <= index < len(results):
        st.info("This result is no longer available.")
        return
    item, r = images[index], results[index]
    st.caption(f"{item['name']}  ({index + 1} of {len(results)})")

    img_col, info_col = st.columns([3, 2], gap="large")
    with img_col:
        st.image(make_preview(item["bytes"]), use_container_width=True)
    with info_col:
        if r["recognized"]:
            st.subheader(pretty(r["label"]).upper())
        else:
            st.subheader("NOT RECOGNIZED")
        st.write(f"Confidence: **{r['confidence']:.1f}%**")
        st.progress(min(max(r["confidence"] / 100, 0.0), 1.0))

        if r["recognized"]:
            st.info(ACTIONS.get(r["label"], DEFAULT_ACTION))
        else:
            st.warning(f"Not sure (best guess: {pretty(r['best_guess'])}, below the "
                       f"{r['threshold']:.0f}% threshold). Try a clearer, well-lit close-up.")

        if st.session_state.show_top3:
            st.write("**Top 3**")
            for name, pct in r["top3"]:
                st.write(f"{pretty(name)}: {pct:.1f}%")


def short_name(name: str, limit: int = 18) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def show_errors():
    errors = st.session_state.upload_errors
    if errors:
        with st.container(height=min(60 * len(errors), 140), border=False):
            for error in errors:
                st.error(error)


def dropzone_css():
    """Big, roomy drop zone for the upload step."""
    st.markdown(
        """
        <style>
        [data-testid="stFileUploaderDropzone"] {
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            gap: 1.5rem;
            padding: 4rem 1rem;
            border: 2px dashed rgba(128, 128, 128, 0.6);
            border-radius: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tight_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 1rem; }

        /* Hide Streamlit's built-in "fullscreen" button that appears when hovering an image */
        [data-testid="stImageContainer"] button,
        [data-testid="StyledFullScreenButton"],
        button[title="View fullscreen"] { display: none !important; }

        /* Thumbnail card: View / Remove float over the picture and appear on hover */
        [class*="st-key-gthumb_"] { position: relative; }
        [class*="st-key-gthumb_"]::after {          /* soft dark tint behind the buttons */
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 0.5rem;
            background: rgba(0, 0, 0, 0.35);
            opacity: 0;
            transition: opacity 0.15s ease;
            pointer-events: none;
            z-index: 2;
        }
        [class*="st-key-gview_"],
        [class*="st-key-gremove_"] {
            position: absolute !important;
            left: 14%;
            width: 72% !important;
            z-index: 3;
            opacity: 0;
            transition: opacity 0.15s ease;
        }
        [class*="st-key-gview_"]   { top: calc(50% - 40px); }
        [class*="st-key-gremove_"] { top: calc(50% + 4px); }

        [class*="st-key-gview_"] button,
        [class*="st-key-gremove_"] button {
            width: 100%;
            min-height: 36px;
            color: #fff !important;
            background: rgba(20, 20, 20, 0.45);
            border: 1px solid rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(3px);
        }
        [class*="st-key-gview_"] button p,
        [class*="st-key-gremove_"] button p { color: #fff !important; }
        [class*="st-key-gview_"] button:hover,
        [class*="st-key-gremove_"] button:hover {
            background: rgba(20, 20, 20, 0.7);
            border-color: #fff;
        }

        [class*="st-key-gthumb_"]:hover::after,
        [class*="st-key-gthumb_"]:hover [class*="st-key-gview_"],
        [class*="st-key-gthumb_"]:hover [class*="st-key-gremove_"],
        [class*="st-key-gthumb_"]:focus-within [class*="st-key-gview_"],
        [class*="st-key-gthumb_"]:focus-within [class*="st-key-gremove_"] { opacity: 1; }

        /* Touch screens have no hover, so keep the buttons visible there */
        @media (hover: none) {
            [class*="st-key-gview_"],
            [class*="st-key-gremove_"] { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_result_card(index, item, r):
    """Compact result row: same-size thumbnail, label + confidence, and a View button."""
    with st.container(border=True):
        thumb, info, action = st.columns([1, 4, 1], vertical_alignment="top")
        with thumb:
            st.image(make_thumbnail(item["bytes"], 200), use_container_width=True)
        with info:
            st.caption(short_name(item["name"], 40))
            if r["recognized"]:
                st.markdown(f"**{pretty(r['label']).upper()}**")
            else:
                st.markdown("**NOT RECOGNIZED**")
            st.progress(min(max(r["confidence"] / 100, 0.0), 1.0),
                        text=f"Confidence: {r['confidence']:.1f}%")
        with action:
            if st.button("View More", key=f"view_result_{index}", use_container_width=True):
                view_result(index)


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("Classifies the condition of road surfaces from photos.")
    st.caption(f"Supported formats: JPG, JPEG, PNG (max {MAX_FILE_MB} MB each)")
    st.caption("Results are estimates and should not replace a professional inspection.")
    with st.expander("Tips for a good photo"):
        st.write(
            "- Shoot straight down or at a slight angle.\n"
            "- Use daylight and avoid heavy shadows.\n"
            "- Let the road surface fill most of the frame.\n"
            "- Keep the camera steady to avoid blur."
        )

# ---------- Model files must exist ----------
missing = missing_files()
if missing:
    st.error("Model files not found. Copy them to app/models/ and reload:\n\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()

tight_css()

# ---------- Header ----------
st.title("Road Surface Classifier")

# ---------- Which step are we on? ----------
images = st.session_state.images
if st.session_state.results is not None:
    step = "results"
elif not images or st.session_state.adding:
    step = "upload"
else:
    step = "classify"

# =====================================================================
# STEP 1: Upload panel (the only panel shown before any images exist)
# =====================================================================
if step == "upload":
    st.write("Upload one or more images of road surfaces to classify their condition.")
    st.caption("Works best on a close-up of one road area. A photo with several different defects gets one label.")

    with st.container(border=True):
        if images:
            st.markdown(f"**Add more images** ({len(images)} already added)")
        else:
            st.markdown("**Upload Images**")

        dropzone_css()
        st.file_uploader(
            "Drag and drop images here, or browse files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            on_change=on_upload,
        )
        show_errors()

        if images:
            st.button("Back to classification", on_click=cancel_adding, use_container_width=True)

# =====================================================================
# STEP 2: Classification panel (images uploaded, ready to classify)
# =====================================================================
elif step == "classify":
    n = len(images)
    st.write("Review your images, then classify them.")

    with st.container(border=True):
        st.markdown(f"**Classification** ({n} image{'s' if n != 1 else ''} ready)")
        show_errors()

        # Thumbnails live in a fixed-height box: the box scrolls, the page doesn't
        with st.container(height=PANEL_HEIGHT, border=False):
            cols = st.columns(GRID_COLUMNS)
            for i, item in enumerate(images):
                with cols[i % GRID_COLUMNS]:
                    # View / Remove are styled (see tight_css) to float over the picture on hover
                    with st.container(key=f"gthumb_{i}"):
                        st.image(make_thumbnail(item["bytes"]), use_container_width=True)
                        if st.button("View", key=f"gview_{i}"):
                            view_image(i)
                        st.button("Remove", key=f"gremove_{i}", on_click=remove_image, args=(i,))
                    st.caption(short_name(item["name"]))

        classify_col, add_col, clear_col = st.columns([2, 1, 1])
        with classify_col:
            run = st.button("Classify Images", use_container_width=True, type="primary")
        with add_col:
            st.button("Add more images", on_click=start_adding, use_container_width=True)
        with clear_col:
            st.button("Clear all", on_click=reset, use_container_width=True)

    if run:
        with st.spinner("Loading the model (first run only)..."):
            bundle = load_model()                        # cached: loaded once, reused afterwards
        results = []
        progress = st.progress(0.0, text="Analyzing road surfaces...")
        for i, item in enumerate(images, start=1):
            results.append(predict(
                load_image(item["bytes"]),
                threshold=st.session_state.threshold,
                bundle=bundle,
            ))
            progress.progress(i / n, text=f"Analyzing {i} of {n}...")

        # Save to persistent history (shown on the History tab)
        save_history(images, results)

        st.session_state.results = results
        st.rerun()

# =====================================================================
# STEP 3: Results (same panel, now showing the classification output)
# =====================================================================
else:
    results = st.session_state.results
    n = len(results)
    recognized = sum(1 for r in results if r["recognized"])

    with st.container(border=True):
        st.markdown(f"**Classification Results** ({recognized} of {n} recognized)")

        # Results scroll inside a fixed-height box; the button stays visible below it
        with st.container(height=PANEL_HEIGHT + 60, border=False):
            for i, (item, r) in enumerate(zip(images, results)):
                show_result_card(i, item, r)

        st.button("Upload More Images", on_click=new_batch, use_container_width=True, type="primary")