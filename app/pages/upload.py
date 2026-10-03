import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

# ---------- Page setup ----------
st.set_page_config(
    layout="wide",
)

LOW_CONFIDENCE = 60.0  # below this, ask for a clearer photo

ACTIONS = {
    "POTHOLE": "Consider reporting this to your local roads authority.",
}
DEFAULT_ACTION = "No specific action suggested for this result."

# ---------- State ----------
st.session_state.setdefault("images", [])     # classified images: [{"name", "bytes"}]
st.session_state.setdefault("results", None)  # list of (label, confidence) or None
st.session_state.setdefault("uploader_key", 0)


def classify(image: Image.Image):
    """TODO: replace with your real model. Return (label, confidence_percent)."""
    return "POTHOLE", 94.2


def reset():
    st.session_state.images = []
    st.session_state.results = None
    st.session_state.uploader_key += 1  # gives a fresh, empty uploader


def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


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
    st.caption("Supported formats: JPG, JPEG, PNG")
    st.caption("Results are estimates and should not replace a professional inspection.")

# ---------- Header ----------
st.title("Road Surface Classifier")
st.write("Upload one or more images of road surfaces to classify their condition.")

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

        images, skipped = [], []
        for f in uploaded_files or []:
            data = f.getvalue()
            try:
                Image.open(io.BytesIO(data)).load()  # fails here if corrupt
            except (UnidentifiedImageError, OSError):
                skipped.append(f.name)
            else:
                images.append({"name": f.name, "bytes": data})

        dropzone_css(has_files=bool(images))
        if skipped:
            st.warning("Skipped unreadable file(s): " + ", ".join(skipped))
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
            results = []
            progress = st.progress(0.0, text="Analyzing road surfaces...")
            for i, item in enumerate(images, start=1):
                results.append(classify(load_image(item["bytes"])))
                progress.progress(i / n, text=f"Analyzing {i} of {n}...")
            st.session_state.images = images
            st.session_state.results = results
            st.rerun()

        st.button("Clear all images", on_click=reset, type="tertiary")

    else:
        st.caption("Classification Results")

        for item, (label, confidence) in zip(
            st.session_state.images, st.session_state.results
        ):
            with st.container(border=True):
                st.caption(item["name"])
                st.subheader(label)
                st.write(f"Confidence: **{confidence:.1f}%**")
                st.progress(min(max(confidence / 100, 0.0), 1.0))

                if confidence < LOW_CONFIDENCE:
                    st.warning("Not sure. Try a clearer, well-lit photo.")
                else:
                    st.caption(ACTIONS.get(label, DEFAULT_ACTION))

        st.button("Upload More Images", on_click=reset, use_container_width=True)

# ---------- Footer ----------
with st.expander("Tips for a good photo"):
    st.write(
        "- Shoot straight down or at a slight angle.\n"
        "- Use daylight and avoid heavy shadows.\n"
        "- Let the road surface fill most of the frame.\n"
        "- Keep the camera steady to avoid blur."
    )