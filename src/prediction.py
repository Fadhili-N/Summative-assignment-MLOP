"""Single-image prediction using the saved model."""
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.model import MODEL_PATH, load_model
from src.preprocessing import CLASS_NAMES, load_image

_model = None
_fn = None
_mtime = None
_lock = threading.Lock()


def get_model(reload=False):
    """Load the model once and reuse it (loading on every request would be very slow)."""
    global _model
    with _lock:
        if _model is None or reload:
            model = load_model(MODEL_PATH)
            global _fn
            # Compile into one optimised graph: much faster than running layer by layer.
            _fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
            _fn(np.zeros((1, 128, 128, 3), dtype=np.float32))   # warm-up (builds the graph)
            _model = model
            global _mtime
            _mtime = MODEL_PATH.stat().st_mtime
    return _model


def _reload_if_changed():
    """If another container retrained and replaced the model file, pick up the new one."""
    try:
        if MODEL_PATH.stat().st_mtime != _mtime:
            get_model(reload=True)
    except FileNotFoundError:
        pass


def predict(source) -> dict:
    """source: file path, raw bytes or PIL image. Returns label + confidence."""
    arr = load_image(source)[None]                       # shape (1, 128, 128, 3)
    # Calling the model directly is far faster than .predict() for a single image.
    get_model()
    _reload_if_changed()
    p_unin = float(_fn(arr).numpy()[0, 0])   # model outputs P(Uninfected)
    label_idx = int(p_unin >= 0.5)
    return {"label": CLASS_NAMES[label_idx],
            "confidence": round(max(p_unin, 1 - p_unin), 4),
            "p_parasitized": round(1 - p_unin, 4),
            "p_uninfected": round(p_unin, 4)}
