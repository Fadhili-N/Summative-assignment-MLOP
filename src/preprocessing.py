"""Data acquisition + preprocessing for the malaria cell classifier.

Folder layout expected (class name = folder name, alphabetical => Parasitized=0, Uninfected=1):
    <dir>/Parasitized/*.png|jpg
    <dir>/Uninfected/*.png|jpg
"""
import io
import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128
CLASS_NAMES = ["Parasitized", "Uninfected"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_image(source) -> np.ndarray:
    """Open a path, bytes or PIL image and return a (128,128,3) float32 array in 0-255.
    Scaling to [-1, 1] happens INSIDE the model (Rescaling layer), so do not scale here."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32)


def is_valid_image(data: bytes) -> bool:
    """True if the bytes are a readable image (used to reject bad uploads)."""
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def list_labeled_files(root):
    """Return [(path, label_index)] for every image under root/<class_name>/."""
    root = Path(root)
    items = []
    for label, name in enumerate(CLASS_NAMES):
        for f in sorted((root / name).glob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((str(f), label))
    return items


def make_dataset(items, batch_size=32, shuffle=False, seed=42):
    """Build a tf.data pipeline: read -> resize -> batch -> prefetch."""
    paths = [p for p, _ in items]
    labels = [float(l) for _, l in items]

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)
    return ds.map(_load, tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_items(items, val_fraction=0.2, seed=42):
    """Shuffle and split a list of (path, label) into train / validation."""
    items = list(items)
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * val_fraction)) if len(items) > 4 else 0
    return items[n_val:], items[:n_val]


def save_image_bytes(data: bytes, dest_dir, cls: str, ext: str = ".png"):
    """Validate image bytes and store them as dest_dir/<cls>/<random name>. Returns the path,
    or None if the bytes are not a readable image. The name is generated, never user-supplied."""
    if cls not in CLASS_NAMES or ext.lower() not in IMAGE_EXTS or not is_valid_image(data):
        return None
    out_dir = Path(dest_dir) / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(__import__('time').time())}_{os.urandom(4).hex()}{ext.lower()}"
    out_path.write_bytes(data)
    return str(out_path)


def save_uploaded_zip(zip_bytes: bytes, dest_dir, label_names=CLASS_NAMES):
    """Extract an uploaded zip of images into dest_dir/<class>/ and return counts + saved paths.

    The zip must contain folders named Parasitized/ and Uninfected/ (any depth).
    Non-image and unreadable files are skipped, never trusted. File names are
    regenerated so a malicious zip cannot write outside dest_dir.
    """
    dest_dir = Path(dest_dir)
    counts = {n: 0 for n in label_names}
    saved_files = []
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = [p.lower() for p in Path(info.filename).parts]
            cls = next((n for n in label_names if n.lower() in parts[:-1]), None)
            ext = Path(info.filename).suffix.lower()
            if cls is None or ext not in IMAGE_EXTS or info.file_size > 5_000_000:
                skipped += 1
                continue
            path = save_image_bytes(z.read(info), dest_dir, cls, ext)
            if path is None:
                skipped += 1
                continue
            saved_files.append((path, label_names.index(cls)))
            counts[cls] += 1
    return {"saved": counts, "skipped": skipped, "files": saved_files}
