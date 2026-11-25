"""FastAPI backend: prediction, bulk upload, retraining, monitoring, and the web UI.

Run from the project root:  uvicorn app.main:app --port 8000
"""
import json
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app import db
from src.model import INFO_PATH, retrain
from src.prediction import get_model, predict
from src.preprocessing import CLASS_NAMES, list_labeled_files, save_image_bytes, save_uploaded_zip

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "data" / "uploads"))
INSIGHTS_PATH = ROOT / "models" / "insights.json"
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", 50))   # auto-retrain after this many new images
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")                    # if set, upload/retrain need this header
MAX_UPLOAD_BYTES = 100_000_000
MAX_IMAGE_BYTES = 5_000_000

STARTED_AT = time.time()
retrain_state = {"running": False, "last": None}
_retrain_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    db.init()
    db.start_flusher()
    get_model()   # load the model once at startup so the first request is fast
    yield
    db.flush()


app = FastAPI(title="Malaria Cell Classifier", lifespan=lifespan)


def _data_dir(split):
    """Full dataset if present (local), otherwise the small committed sample (Docker/cloud)."""
    full = ROOT / "data" / split
    return full if list_labeled_files(full) else ROOT / "data" / "sample" / split


def _check_token(token):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(401, "Missing or wrong admin token.")


# ---------- prediction ----------
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB).")
    from src.preprocessing import is_valid_image
    if not is_valid_image(data):
        raise HTTPException(400, "That file is not a readable image.")
    start = time.perf_counter()
    result = await run_in_threadpool(predict, data)   # keep the server responsive while the model runs
    ms = (time.perf_counter() - start) * 1000
    db.log_prediction(result["label"], result["confidence"], ms)
    return {**result, "latency_ms": round(ms, 1)}


# ---------- upload + retraining ----------
@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...), label: str = Form(""),
                     x_admin_token: str = Header("")):
    """Accepts a .zip (folders Parasitized/ and Uninfected/) or several images + a chosen label."""
    _check_token(x_admin_token)
    saved, skipped = [], 0
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{f.filename} is too large.")
        name = (f.filename or "").lower()
        if name.endswith(".zip"):
            try:
                res = save_uploaded_zip(data, UPLOAD_DIR)
            except Exception:
                raise HTTPException(400, f"{f.filename} is not a valid zip file.")
            saved += res["files"]; skipped += res["skipped"]
        else:
            if label not in CLASS_NAMES:
                raise HTTPException(400, "For image uploads, choose a label (Parasitized or Uninfected).")
            ext = os.path.splitext(name)[1] or ".png"
            path = save_image_bytes(data, UPLOAD_DIR, label, ext) if len(data) <= MAX_IMAGE_BYTES else None
            if path: saved.append((path, CLASS_NAMES.index(label)))
            else: skipped += 1
    if not saved:
        raise HTTPException(400, f"No valid images found ({skipped} files skipped).")
    db.add_uploads(saved)
    counts = db.upload_counts()
    auto = False
    if counts["pending"] >= RETRAIN_THRESHOLD:
        auto = _start_retrain()
    return {"saved": len(saved), "skipped": skipped, "pending": counts["pending"],
            "auto_retrain_started": auto, "threshold": RETRAIN_THRESHOLD}


def _start_retrain(epochs=3):
    with _retrain_lock:
        if retrain_state["running"]:
            return False
        retrain_state["running"] = True
    threading.Thread(target=_do_retrain, args=(epochs,), daemon=True).start()
    return True


def _do_retrain(epochs):
    try:
        rows = db.pending_uploads()
        new_items = [(p, l) for p, l, _ in rows if os.path.exists(p)]
        # Replay: mix in some original training images so the model does not forget old data.
        old = list_labeled_files(_data_dir("train"))
        replay = random.Random(0).sample(old, min(len(old), max(len(new_items), 20), 500)) if old else []
        test_files = list_labeled_files(_data_dir("test"))
        reference = random.Random(1).sample(test_files, min(len(test_files), 200)) if test_files else None
        report = retrain(new_items + replay, epochs=epochs, reference_items=reference)
        report["new_images"] = len(new_items); report["replay_images"] = len(replay)
        if report["accepted"]:
            get_model(reload=True)
        db.mark_used([i for _, _, i in rows])
        db.log_retrain(len(new_items), report["accepted"], report)
        retrain_state["last"] = {"ok": True, **report}
    except Exception as e:
        retrain_state["last"] = {"ok": False, "error": str(e)}
    finally:
        retrain_state["running"] = False


@app.post("/api/retrain")
def api_retrain(epochs: int = 3, x_admin_token: str = Header("")):
    _check_token(x_admin_token)
    pending = db.upload_counts()["pending"]
    if pending == 0:
        raise HTTPException(400, "No new uploaded images are waiting. Upload data first.")
    if not _start_retrain(max(1, min(epochs, 10))):
        raise HTTPException(409, "A retraining job is already running.")
    return {"started": True, "pending": pending}


@app.get("/api/retrain/status")
def api_retrain_status():
    return {"running": retrain_state["running"], "last": retrain_state["last"],
            "pending": db.upload_counts()["pending"], "threshold": RETRAIN_THRESHOLD,
            "history": db.retrain_history(5)}


# ---------- monitoring + insights ----------
@app.get("/api/health")
def api_health():
    info = json.loads(INFO_PATH.read_text()) if INFO_PATH.exists() else {}
    hist = db.retrain_history(100)
    return {"status": "ok", "uptime_seconds": round(time.time() - STARTED_AT),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(STARTED_AT)),
            "model_loaded": True, "model_version": 1 + sum(1 for h in hist if h["accepted"]),
            "base_model": info.get("best_experiment"), "test_metrics": info.get("test_metrics"),
            "retrain_threshold": RETRAIN_THRESHOLD}


@app.get("/api/monitor")
def api_monitor():
    rows = db.prediction_stats()
    lat = [r["latency_ms"] for r in rows]
    conf = [r["confidence"] for r in rows]
    hist, _ = np.histogram(conf, bins=10, range=(0.5, 1.0)) if conf else (np.zeros(10, int), None)
    return {"total_predictions": len(rows),
            "by_label": {n: sum(1 for r in rows if r["label"] == n) for n in CLASS_NAMES},
            "latency_ms": {"mean": round(float(np.mean(lat)), 1) if lat else None,
                           "p95": round(float(np.percentile(lat, 95)), 1) if lat else None},
            "confidence_hist": hist.tolist(),
            "low_confidence_share": round(sum(c < 0.7 for c in conf) / len(conf), 3) if conf else None,
            "uploads": db.upload_counts()}


@app.get("/api/insights")
def api_insights():
    return json.loads(INSIGHTS_PATH.read_text()) if INSIGHTS_PATH.exists() else {"features": []}


app.mount("/", StaticFiles(directory=str(ROOT / "app" / "static"), html=True), name="static")
