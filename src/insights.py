"""Compute simple, explainable image features per class and save them for the UI charts.

Usage:  python -m src.insights            (needs data/train from src.data_acquisition)
Writes models/insights.json. Interpretations live in models/insight_stories.json
(key = feature key) so they can be written after looking at the numbers.
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import CLASS_NAMES, IMAGE_EXTS

ROOT = Path(__file__).resolve().parent.parent
N_PER_CLASS = 1500
BG_THRESHOLD = 25   # pixels darker than this are the black slide background, not the cell

FEATURES = {
    "cell_area":   ("Cell size", "share of the image the cell fills"),
    "brightness":  ("Brightness", "mean grey level of the cell (0-255)"),
    "saturation":  ("Colour saturation", "how strongly coloured the cell is (0-1)"),
    "texture":     ("Texture / contrast", "std of grey levels inside the cell"),
    "image_pixels": ("Raw image size", "width x height in pixels (thousands)"),
}


def image_features(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    small = img.resize((128, 128))
    arr = np.asarray(small, dtype=np.float32)
    gray = arr.mean(axis=2)
    cell = gray > BG_THRESHOLD
    if cell.sum() < 50:
        cell = np.ones_like(cell, dtype=bool)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    return {"cell_area": float(cell.mean()), "brightness": float(gray[cell].mean()),
            "saturation": float(sat[cell].mean()), "texture": float(gray[cell].std()),
            "image_pixels": w * h / 1000}


def main():
    rng = random.Random(42)
    vals = {k: {c: [] for c in CLASS_NAMES} for k in FEATURES}
    for cls in CLASS_NAMES:
        files = [f for f in (ROOT / "data" / "train" / cls).glob("*") if f.suffix.lower() in IMAGE_EXTS]
        for f in rng.sample(files, min(N_PER_CLASS, len(files))):
            for k, v in image_features(f).items():
                vals[k][cls].append(v)

    stories_path = ROOT / "models" / "insight_stories.json"
    stories = json.loads(stories_path.read_text()) if stories_path.exists() else {}
    out = {"n_per_class": {c: len(vals["brightness"][c]) for c in CLASS_NAMES}, "features": []}
    for k, (title, desc) in FEATURES.items():
        pooled = np.concatenate([vals[k][c] for c in CLASS_NAMES])
        lo, hi = np.percentile(pooled, [1, 99])
        edges = np.linspace(lo, hi, 21)
        entry = {"key": k, "title": title, "description": desc, "edges": [round(float(e), 4) for e in edges],
                 "classes": {}, "story": stories.get(k, "")}
        for c in CLASS_NAMES:
            a = np.array(vals[k][c])
            counts, _ = np.histogram(np.clip(a, lo, hi), bins=edges)
            entry["classes"][c] = {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                                   "median": round(float(np.median(a)), 4),
                                   "hist": (counts / counts.sum()).round(4).tolist()}
        a, b = np.array(vals[k][CLASS_NAMES[0]]), np.array(vals[k][CLASS_NAMES[1]])
        pooled_sd = np.sqrt((a.var() + b.var()) / 2)
        entry["effect_size"] = round(float((a.mean() - b.mean()) / pooled_sd), 3) if pooled_sd else 0.0
        out["features"].append(entry)

    (ROOT / "models" / "insights.json").write_text(json.dumps(out, indent=1))
    for f in out["features"]:
        print(f"{f['title']:20s} " + "  ".join(f"{c}: {f['classes'][c]['mean']:.3f}" for c in CLASS_NAMES)
              + f"   effect size d = {f['effect_size']}")


if __name__ == "__main__":
    main()
