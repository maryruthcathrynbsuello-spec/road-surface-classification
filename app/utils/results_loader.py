"""Reads the real model results from docs/ (no mock data).

Put this file in app/utils/. Both app/views/dashboard.py and
app/views/model_performance.py import from it, so every number on the
two pages comes from the same metrics files.
"""
import json
import re
from collections import deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures"
REPORT_DIR = ROOT / "docs" / "data_reports"

MODEL_NAME = "efficientnetb0_v4"          
HISTORY_FILE = ROOT / "app" / "views" / "history.json"   

CLASSES = ["crack", "damaged_asphalt", "normal", "open_manhole",
           "pothole", "water_filled_pothole"]
MODEL_LABELS = {"baseline_cnn": "Baseline CNN", "mobilenetv2": "MobileNetV2",
                "efficientnetb0": "EfficientNetB0"}


def norm(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


CLASS_SET = {norm(c) for c in CLASSES}


def pretty(name):
    return str(name).replace("_", " ")


def find_figure(*patterns):
    for pattern in patterns:
        matches = sorted(FIG_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_json(path):
    """Return (data, error). data is None if the file is missing or broken."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (json.JSONDecodeError, OSError) as e:
        return None, f"Could not read {path.name}: {e}"


def model_label(stem):
    """metrics_mobilenetv2_v3 -> 'MobileNetV2 v3'."""
    stem = stem.replace("metrics_", "", 1)
    m = re.match(r"(.+)_v(\d+)$", stem)
    base, ver = (m.group(1), m.group(2)) if m else (stem, None)
    label = MODEL_LABELS.get(base, pretty(base).title())
    return f"{label} v{ver}" if ver else label


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def find_value(d, names):
    """First numeric value (shallowest first) whose key matches one of names."""
    names = {norm(n) for n in names}
    queue = deque([d])
    while queue:
        cur = queue.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if norm(k) in names and _num(v):
                    return float(v)
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(x for x in cur if isinstance(x, (dict, list)))
    return None


def pick_split(d, kind):
    """If the file separates splits (test / val), return that part."""
    names = {"test": {"test", "test_metrics", "test_set"},
             "val": {"val", "validation", "valid", "val_metrics"}}[kind]
    if isinstance(d, dict):
        for k, v in d.items():
            if norm(k) in names and isinstance(v, dict):
                return v
    return d


def _macro_from_avg(d):
    queue = deque([d])
    while queue:
        cur = queue.popleft()
        if isinstance(cur, dict):
            for k, val in cur.items():
                if norm(k) in {"macro_avg", "macro"} and isinstance(val, dict):
                    return find_value(val, ["f1_score", "f1"])
            queue.extend(v for v in cur.values() if isinstance(v, dict))
    return None


def scores(data, kind):
    """Return (accuracy, macro_f1, label). label says which split the numbers
    are from: 'Test', 'Validation', or '' when the file does not say."""
    split = pick_split(data, kind)
    in_split = split is not data
    names = {
        "test": (["test_accuracy", "test_acc"], ["test_macro_f1", "macro_f1_test"]),
        "val": (["val_accuracy", "val_acc"], ["val_macro_f1", "macro_f1_val"]),
    }[kind]
    wanted = "Test" if kind == "test" else "Validation"
    acc = find_value(split, names[0])
    f1 = find_value(split, names[1])
    if acc is not None or f1 is not None:
        return acc, f1, wanted
    acc = find_value(split, ["accuracy", "acc"])
    f1 = find_value(split, ["macro_f1", "f1_macro"])
    if f1 is None:
        f1 = _macro_from_avg(split)
    if acc is not None or f1 is not None:
        return acc, f1, (wanted if in_split else "")
    if kind == "test":
        acc = find_value(data, ["val_accuracy", "val_acc"])
        f1 = find_value(data, ["val_macro_f1", "macro_f1_val"])
        if acc is not None or f1 is not None:
            return acc, f1, "Validation"
    return None, None, ""


def per_class(d):
    """Per-class precision/recall/F1/support as a DataFrame, or None."""
    queue = deque([d])
    while queue:
        cur = queue.popleft()
        if isinstance(cur, dict):
            num_hits = [k for k in cur if norm(k) in CLASS_SET and _num(cur[k])]
            if len(num_hits) >= 3:
                rows = {norm(k): {"F1": cur[k]} for k in num_hits}
                df = pd.DataFrame(rows).T.reindex(
                    columns=["Test images", "Precision", "Recall", "F1"])
                order = [c for c in CLASSES if c in df.index]
                df = df.loc[order]
                df.index = [pretty(c) for c in df.index]
                df.index.name = "Class"
                return df
            hits = [k for k in cur if norm(k) in CLASS_SET and isinstance(cur[k], dict)]
            if len(hits) >= 3:
                rows = {}
                for k in hits:
                    v = {norm(a): b for a, b in cur[k].items()}
                    rows[norm(k)] = {
                        "Test images": v.get("support", v.get("n")),
                        "Precision": v.get("precision"),
                        "Recall": v.get("recall"),
                        "F1": v.get("f1_score", v.get("f1")),
                    }
                df = pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")
                order = [c for c in CLASSES if c in df.index]
                df = df.loc[order]
                df.index = [pretty(c) for c in df.index]
                df.index.name = "Class"
                return df
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(x for x in cur if isinstance(x, (dict, list)))
    return None


def training_setup(d):
    keys = ["backbone", "epochs_head", "epochs_fine", "fine_tune_layers",
            "lr_head", "lr_fine", "batch_size", "params"]
    if not isinstance(d, dict):
        return {}
    return {k: d[k] for k in keys if k in d}


def setup_text(setup):
    parts = []
    if "backbone" in setup:
        parts.append(f"Backbone: {setup['backbone']}")
    if _num(setup.get("epochs_head")) and _num(setup.get("epochs_fine")):
        parts.append(f"{setup['epochs_head']} head epochs + "
                     f"{setup['epochs_fine']} fine-tuning epochs")
    if "fine_tune_layers" in setup:
        parts.append(f"{setup['fine_tune_layers']} layers fine-tuned")
    if "batch_size" in setup:
        parts.append(f"batch size {setup['batch_size']}")
    if _num(setup.get("params")):
        parts.append(f"{int(setup['params']):,} parameters")
    return " · ".join(parts)


def summary(path, kind="test"):
    """Read one metrics file and pull out the numbers the pages show."""
    data, err = read_json(path)
    out = {"data": data, "error": err, "accuracy": None, "macro_f1": None,
           "label": "", "per_class": None, "total_images": None,
           "setup": {}, "epochs_total": None}
    if data is None:
        return out
    out["accuracy"], out["macro_f1"], out["label"] = scores(data, kind)
    out["per_class"] = per_class(pick_split(data, kind))
    if out["per_class"] is None:
        out["per_class"] = per_class(data)
    pc = out["per_class"]
    if pc is not None and pc["Test images"].notna().any():
        out["total_images"] = int(pc["Test images"].sum())
    out["setup"] = training_setup(data)
    st_ = out["setup"]
    if _num(st_.get("epochs_head")) and _num(st_.get("epochs_fine")):
        out["epochs_total"] = int(st_["epochs_head"] + st_["epochs_fine"])
    return out


def metrics_path(model_name=MODEL_NAME):
    return REPORT_DIR / f"metrics_{model_name}.json"


def comparison_table():
    """One row per metrics_*.json file: accuracy and macro F1 (validation split)."""
    rows = {}
    for path in sorted(REPORT_DIR.glob("metrics_*.json")):
        s = summary(path, "val")
        if s["accuracy"] is not None and s["macro_f1"] is not None:
            rows[model_label(path.stem)] = {"Accuracy": s["accuracy"],
                                            "Macro F1": s["macro_f1"]}
    if not rows:
        return None
    return pd.DataFrame(rows).T.sort_values("Macro F1", ascending=False)


HISTORY_CLASS_COLUMNS = ["predicted_class", "prediction", "predicted_label",
                         "predicted", "pred_class", "class", "label",
                         "result", "top_class"]


def load_prediction_history(path=HISTORY_FILE):
    """Return (DataFrame or None, class_column or None, error or None)."""
    path = Path(path)
    if not path.exists():
        return None, None, None
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("history", "predictions", "records", "items", "data"):
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
                else:
                    data = [data] if data else []
            df = pd.json_normalize(data)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        return None, None, f"Could not read {path.name}: {e}"
    if df.empty:
        return None, None, None
    for col in df.columns:
        if norm(str(col).split(".")[-1]) in HISTORY_CLASS_COLUMNS:
            return df, col, None
    return df, None, None