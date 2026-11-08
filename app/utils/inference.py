"""CNN-only inference: one uploaded photo in, one predicted class (plus top 3) out.

Place at app/utils/inference.py (next to an empty __init__.py).
Model files go in app/models/ (do not commit them to Git):
    efficientnetb0_v2.keras, preprocessing_config.json
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:                                    # inside Streamlit the model is cached; elsewhere it is a plain call
    import streamlit as st
    _cache = st.cache_resource
except Exception:                       # pragma: no cover
    _cache = lambda f: f

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / 'models'
MODEL_PATH = MODEL_DIR / 'efficientnetb0_v2.keras'
CONFIG_PATH = MODEL_DIR / 'preprocessing_config.json'

# ---------- easy-to-change settings ----------
DEFAULT_THRESHOLD = 0.60                # below this the app says "Not recognized"
MAX_FILE_MB = 10                        # keep in sync with .streamlit/config.toml (maxUploadSize)
ALLOWED_EXTENSIONS = ('jpg', 'jpeg', 'png')
MIN_SIDE_PX = 32                        # the model was trained on regions of at least this size
NOT_RECOGNIZED = 'Not recognized'


def missing_files():
    return [str(p) for p in (MODEL_PATH, CONFIG_PATH) if not p.exists()]


def pretty(name: str) -> str:
    return name.replace('_', ' ')


@_cache
def load_model():
    """Loads the model once. Returns (model, class_names, img_size)."""
    from tensorflow import keras         # imported here so the page opens fast
    cfg = json.load(open(CONFIG_PATH))
    class_names, img_size = cfg['class_names'], int(cfg['img_size'])
    model = keras.models.load_model(MODEL_PATH)
    if model.output_shape[-1] != len(class_names):
        raise ValueError(f'Model has {model.output_shape[-1]} outputs but the config lists {len(class_names)} classes.')
    return model, class_names, img_size


def validate_upload(name: str, data: bytes):
    """Returns (PIL image, None) if the upload is usable, otherwise (None, clear error message)."""
    ext = Path(name).suffix.lower().lstrip('.')
    if ext not in ALLOWED_EXTENSIONS:
        return None, f'{name}: unsupported file type ".{ext}". Use JPG, JPEG or PNG.'
    if len(data) == 0:
        return None, f'{name}: the file is empty.'
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f'{name}: file is {len(data) / 1e6:.1f} MB, larger than the {MAX_FILE_MB} MB limit.'
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                               # catches truncated and corrupted files
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError, Image.DecompressionBombError):
        return None, f'{name}: this is not a readable image. The file may be corrupted or not an image.'
    if img.format not in ('JPEG', 'MPO', 'PNG'):          # e.g. a GIF renamed to .jpg
        return None, f'{name}: the file content is {img.format}, not JPEG or PNG.'
    img = ImageOps.exif_transpose(img).convert('RGB')
    if min(img.size) < MIN_SIDE_PX:
        return None, f'{name}: image is too small ({img.width}x{img.height}). Minimum side is {MIN_SIDE_PX} px.'
    return img, None


def preprocess(image: Image.Image, img_size: int) -> np.ndarray:
    """Same as training: stretch to img_size x img_size (INTER_AREA, no padding), raw 0-255 pixels.
    EfficientNetB0 rescales inside the model, so do NOT divide by 255 here."""
    arr = np.asarray(image.convert('RGB'))
    arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA).astype('float32')
    return arr[None]


def predict(image: Image.Image, threshold: float = DEFAULT_THRESHOLD, bundle=None):
    """The one prediction function. `bundle` lets tests pass a fake (model, class_names, img_size).

    Returns a dict:
      label       predicted class, or "Not recognized" if confidence < threshold
      best_guess  the top class even when it is below the threshold
      confidence  top confidence in percent
      recognized  True when confidence >= threshold
      top3        [(class_name, percent), ...]
    """
    model, class_names, img_size = bundle or load_model()
    probs = np.asarray(model.predict(preprocess(image, img_size), verbose=0))[0]
    order = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[int(i)], float(probs[i]) * 100) for i in order]
    best, conf = top3[0]
    recognized = conf >= threshold * 100
    return {'label': best if recognized else NOT_RECOGNIZED, 'best_guess': best, 'confidence': conf,
            'recognized': recognized, 'threshold': threshold * 100, 'top3': top3}