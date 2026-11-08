"""Applies the CNN-integration edits to app/views/upload.py and app/views/history.py.

Run from the repository root:   python apply_cnn_changes.py
Safe to run twice. Each edit is reported as: applied / already applied / NOT FOUND.
If any edit in a file is NOT FOUND, that file is left untouched.
Undo with:   git restore app/views/upload.py app/views/history.py
Line endings (Windows CRLF or LF) are preserved. Delete this script afterwards.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

# (label, text to find, replacement, test that tells the edit is already in the file)
EDITS = {'app/views/history.py': [('h1',
                           '        "Detections": [record["detections"] for record in filtered],\n',
                           '        "Confidence": [\n'
                           '            f"{record[\'confidence\']:.1f}%" if record.get("confidence") is not '
                           'None else ""\n'
                           '            for record in filtered\n'
                           '        ],\n')],
 'app/views/upload.py': [('u1',
                          '                "detections": 1,\n',
                          '                "confidence": round(conf, 1),\n'),
                         ('imports',
                          'import io\n'
                          'import json\n'
                          'import uuid\n'
                          'from datetime import datetime\n'
                          'from pathlib import Path\n'
                          '\n'
                          'import streamlit as st\n'
                          'from PIL import Image, UnidentifiedImageError\n',
                          'import io\n'
                          'import json\n'
                          'import sys\n'
                          'import uuid\n'
                          'from datetime import datetime\n'
                          'from pathlib import Path\n'
                          '\n'
                          'import streamlit as st\n'
                          'from PIL import Image, ImageOps\n'
                          '\n'
                          'sys.path.append(str(Path(__file__).resolve().parents[1]))   # app/ folder, so '
                          '`utils` can be imported\n'
                          'from utils.inference import (DEFAULT_THRESHOLD, MAX_FILE_MB, load_model, '
                          'missing_files,\n'
                          '                             predict, pretty, validate_upload)\n'),
                         ('actions',
                          'LOW_CONFIDENCE = 60.0  # below this, ask for a clearer photo\n'
                          '\n'
                          'ACTIONS = {\n'
                          '    "POTHOLE": "Consider reporting this to your local roads authority.",\n'
                          '}\n',
                          'ACTIONS = {\n'
                          '    "pothole": "Consider reporting this to your local roads authority.",\n'
                          '    "water-filled_pothole": "The depth is hidden by the water. Treat it like a '
                          'pothole and consider reporting it.",\n'
                          '    "open_manhole": "An open manhole is a safety hazard. Report it promptly.",\n'
                          '    "crack": "Cracks can be monitored. Report them if they are wide or '
                          'spreading.",\n'
                          '    "damaged_asphalt": "Consider reporting damaged pavement to your local roads '
                          'authority.",\n'
                          '    "normal": "No defect recognized in this photo.",\n'
                          '}\n'),
                         ('state comment',
                          '# list of (label, confidence) or None',
                          '# list of predict() result dicts, or None'),
                         ('classify stub',
                          'def classify(image: Image.Image):\n'
                          '    """TODO: replace with your real model. Return (label, '
                          'confidence_percent)."""\n'
                          '    return "POTHOLE", 94.2\n'
                          '\n'
                          '\n',
                          ''),
                         ('save_history',
                          '    for item, (label, conf) in zip(images, results):\n'
                          '        history.append(\n'
                          '            {\n'
                          '                "id": uuid.uuid4().hex,\n'
                          '                "time": now,\n'
                          '                "file_name": item["name"],\n'
                          '                "confidence": round(conf, 1),\n'
                          '                "labels": [label],\n'
                          '                "status": "Needs review" if conf < LOW_CONFIDENCE else "OK",\n'
                          '            }\n'
                          '        )\n',
                          '    for item, r in zip(images, results):\n'
                          '        history.append(\n'
                          '            {\n'
                          '                "id": uuid.uuid4().hex,\n'
                          '                "time": now,\n'
                          '                "file_name": item["name"],\n'
                          '                "confidence": round(r["confidence"], 1),\n'
                          '                "labels": [r["label"]],\n'
                          '                "status": "OK" if r["recognized"] else "Needs review",\n'
                          '            }\n'
                          '        )\n'),
                         ('load_image',
                          '    return Image.open(io.BytesIO(data)).convert("RGB")\n',
                          '    return '
                          'ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")\n'),
                         ('sidebar+header',
                          '    st.caption("Supported formats: JPG, JPEG, PNG")\n'
                          '    st.caption("Results are estimates and should not replace a professional '
                          'inspection.")\n'
                          '\n'
                          '# ---------- Header ----------\n'
                          'st.title("Road Surface Classifier")\n'
                          'st.write("Upload one or more images of road surfaces to classify their '
                          'condition.")\n',
                          '    st.caption(f"Supported formats: JPG, JPEG, PNG (max {MAX_FILE_MB} MB each)")\n'
                          '    st.caption("Results are estimates and should not replace a professional '
                          'inspection.")\n'
                          '    st.header("Settings")\n'
                          '    threshold = st.slider("Confidence threshold", 0.10, 0.95, DEFAULT_THRESHOLD, '
                          '0.05,\n'
                          '                          help="Below this confidence the app answers \'Not '
                          'recognized\' instead of a label.")\n'
                          '\n'
                          '# ---------- Model files must exist ----------\n'
                          'missing = missing_files()\n'
                          'if missing:\n'
                          '    st.error("Model files not found. Copy them to app/models/ and reload:\\n\\n" '
                          '+ "\\n".join(f"- {m}" for m in missing))\n'
                          '    st.stop()\n'
                          '\n'
                          '# ---------- Header ----------\n'
                          'st.title("Road Surface Classifier")\n'
                          'st.write("Upload one or more images of road surfaces to classify their '
                          'condition.")\n'
                          'st.caption("Works best on a close-up of one road area. A photo with several '
                          'different defects gets one label.")\n'),
                         ('uploader loop',
                          '        images, skipped = [], []\n'
                          '        for f in uploaded_files or []:\n'
                          '            data = f.getvalue()\n'
                          '            try:\n'
                          '                Image.open(io.BytesIO(data)).load()  # fails here if corrupt\n'
                          '            except (UnidentifiedImageError, OSError):\n'
                          '                skipped.append(f.name)\n'
                          '            else:\n'
                          '                images.append({"name": f.name, "bytes": data})\n'
                          '\n'
                          '        dropzone_css(has_files=bool(images))\n'
                          '        if skipped:\n'
                          '            st.warning("Skipped unreadable file(s): " + ", ".join(skipped))\n'
                          '        show_grid(images)\n',
                          '        images, errors = [], []\n'
                          '        for f in uploaded_files or []:\n'
                          '            data = f.getvalue()\n'
                          '            _, error = validate_upload(f.name, data)   # type, size, corrupted or '
                          'non-image checks\n'
                          '            if error:\n'
                          '                errors.append(error)\n'
                          '            else:\n'
                          '                images.append({"name": f.name, "bytes": data})\n'
                          '\n'
                          '        dropzone_css(has_files=bool(images))\n'
                          '        for error in errors:\n'
                          '            st.error(error)\n'
                          '        show_grid(images)\n'),
                         ('classify loop',
                          '            results = []\n'
                          '            progress = st.progress(0.0, text="Analyzing road surfaces...")\n'
                          '            for i, item in enumerate(images, start=1):\n'
                          '                results.append(classify(load_image(item["bytes"])))\n'
                          '                progress.progress(i / n, text=f"Analyzing {i} of {n}...")\n',
                          '            with st.spinner("Loading the model (first run only)..."):\n'
                          '                bundle = load_model()                        # cached: loaded '
                          'once, reused afterwards\n'
                          '            results = []\n'
                          '            progress = st.progress(0.0, text="Analyzing road surfaces...")\n'
                          '            for i, item in enumerate(images, start=1):\n'
                          '                results.append(predict(load_image(item["bytes"]), '
                          'threshold=threshold, bundle=bundle))\n'
                          '                progress.progress(i / n, text=f"Analyzing {i} of {n}...")\n'),
                         ('results display',
                          '        for item, (label, confidence) in zip(\n'
                          '            st.session_state.images, st.session_state.results\n'
                          '        ):\n'
                          '            with st.container(border=True):\n'
                          '                st.caption(item["name"])\n'
                          '                st.subheader(label)\n'
                          '                st.write(f"Confidence: **{confidence:.1f}%**")\n'
                          '                st.progress(min(max(confidence / 100, 0.0), 1.0))\n'
                          '\n'
                          '                if confidence < LOW_CONFIDENCE:\n'
                          '                    st.warning("Not sure. Try a clearer, well-lit photo.")\n'
                          '                else:\n'
                          '                    st.caption(ACTIONS.get(label, DEFAULT_ACTION))\n',
                          '        for item, r in zip(st.session_state.images, st.session_state.results):\n'
                          '            with st.container(border=True):\n'
                          '                st.caption(item["name"])\n'
                          '                if r["recognized"]:\n'
                          '                    st.subheader(pretty(r["label"]).upper())\n'
                          '                else:\n'
                          '                    st.subheader("NOT RECOGNIZED")\n'
                          '                st.write(f"Confidence: **{r[\'confidence\']:.1f}%**")\n'
                          '                st.progress(min(max(r["confidence"] / 100, 0.0), 1.0))\n'
                          '\n'
                          '                if r["recognized"]:\n'
                          '                    st.caption(ACTIONS.get(r["label"], DEFAULT_ACTION))\n'
                          '                else:\n'
                          '                    st.warning(f"Not sure (best guess: '
                          '{pretty(r[\'best_guess\'])}, below the "\n'
                          '                               f"{r[\'threshold\']:.0f}% threshold). Try a '
                          'clearer, well-lit close-up.")\n'
                          '\n'
                          '                st.write("**Top 3**")\n'
                          '                for name, pct in r["top3"]:\n'
                          '                    st.write(f"{pretty(name)}: {pct:.1f}%")\n')]}


def already_done(label, old, new, t):
    if label == "u1":
        return '"detections": 1' not in t
    if label == "h1":
        return 'record["detections"]' not in t
    if new == "":
        return old.splitlines()[0] not in t
    return new in t


def apply_file(rel, edits):
    path = ROOT / rel
    if not path.exists():
        print(f"{rel}: FILE NOT FOUND (run this from the repository root)")
        return False
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    ok = True
    print(rel + (" (Windows line endings)" if crlf else ""))
    for label, old, new in edits:
        n = text.count(old)
        if n == 1:
            text = text.replace(old, new)
            print(f"  applied          {label}")
        elif n > 1:
            print(f"  AMBIGUOUS        {label}: found {n} matches")
            ok = False
        elif already_done(label, old, new, text):
            print(f"  already applied  {label}")
        else:
            first = old.strip().splitlines()[0][:70]
            print(f"  NOT FOUND        {label}: could not find  {first!r}")
            ok = False
    if not ok:
        print("  -> file left unchanged. Send this output and the current file to your assistant.")
        return False
    out = text.replace("\n", "\r\n") if crlf else text
    if out != raw:
        path.write_bytes(out.encode("utf-8"))
        print("  -> saved")
    else:
        print("  -> nothing to change")
    return True


def main():
    if not (ROOT / "app" / "utils" / "inference.py").exists():
        print("WARNING: app/utils/inference.py is missing. Add it before running the app.\n")
    results = [apply_file(rel, edits) for rel, edits in EDITS.items()]
    print("\nDone." if all(results) else "\nSome edits failed. See above.")
    sys.exit(0 if all(results) else 1)


main()
