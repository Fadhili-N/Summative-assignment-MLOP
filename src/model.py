"""Model creation, retraining (fine-tuning the saved model) and evaluation."""
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.preprocessing import make_dataset, split_items

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "malaria_model.h5"
INFO_PATH = ROOT / "models" / "model_info.json"


def load_model(path=MODEL_PATH):
    return tf.keras.models.load_model(path)


def evaluate(model, ds):
    """Metrics with Parasitized (class 0) as the positive class."""
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    p_unin = model.predict(ds, verbose=0).ravel()
    y_pred = (p_unin >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }
    out["auc"] = roc_auc_score(1 - y_true, 1 - p_unin) if len(set(y_true)) == 2 else None
    return {k: (round(float(v), 4) if v is not None else None) for k, v in out.items()}


def retrain(items, epochs=3, lr=1e-5, batch_size=32, min_images=10, model_path=MODEL_PATH,
            reference_items=None):
    """Fine-tune the SAVED model (the pretrained base) on newly uploaded labelled images.

    items: [(path, label_index)] of the new data. Returns a report dict.
    The old model is backed up first. The new one only replaces it if it is not worse than
    the old one on (a) held-out new images and (b) a fixed reference sample of the original
    test set (reference_items), so a retrain cannot quietly make the model worse overall.
    """
    if len(items) < min_images:
        raise ValueError(f"Need at least {min_images} labelled images to retrain, got {len(items)}.")
    train_items, val_items = split_items(items, val_fraction=0.2)
    if len({l for _, l in train_items}) < 2:
        raise ValueError("Need images from both classes (Parasitized and Uninfected).")

    train_ds = make_dataset(train_items, batch_size, shuffle=True)
    val_ds = make_dataset(val_items, batch_size)

    model = load_model(model_path)                       # custom model = pretrained starting point
    before = evaluate(model, val_ds)
    ref_ds = make_dataset(reference_items, batch_size) if reference_items else None
    ref_before = evaluate(model, ref_ds) if ref_ds is not None else None
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy"])
    start = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                                 restore_best_weights=True)])
    after = evaluate(model, val_ds)

    ref_after = evaluate(model, ref_ds) if ref_ds is not None else None
    accepted = after["f1"] >= before["f1"] - 0.01        # accept unless clearly worse on new data...
    if ref_after is not None:                             # ...or on the fixed reference sample
        accepted = accepted and ref_after["f1"] >= ref_before["f1"] - 0.01
    if accepted:
        backup = Path(model_path).with_name(f"malaria_model_backup_{int(time.time())}.h5")
        shutil.copy(model_path, backup)
        tmp = Path(model_path).with_name("malaria_model_tmp.h5")
        model.save(tmp, include_optimizer=False)   # keep the file small (<10 MB)
        os.replace(tmp, model_path)      # atomic swap: other containers never read a half-written file
        info = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}
        info["last_retrain"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_images": len(items),
                                "before": before, "after": after}
        INFO_PATH.write_text(json.dumps(info, indent=2))
    return {"accepted": accepted, "n_train": len(train_items), "n_val": len(val_items),
            "epochs_run": len(hist.history["loss"]), "seconds": round(time.time() - start, 1),
            "before": before, "after": after,
            "reference_before": ref_before, "reference_after": ref_after}
