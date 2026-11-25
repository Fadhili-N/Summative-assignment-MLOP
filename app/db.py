"""Tiny SQLite layer: uploaded training images, prediction log, retraining history."""
import json
import os
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("DB_PATH", ROOT / "models" / "app.db"))


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


# Prediction logs are buffered in memory and written in batches by a background thread,
# so the request path never waits on the database (matters with several containers).
_buf = []
_buf_lock = threading.Lock()


def flush():
    with _buf_lock:
        rows, _buf[:] = list(_buf), []
    if rows:
        with closing(_conn()) as c:
            c.executemany("INSERT INTO predictions(ts,label,confidence,latency_ms) VALUES (?,?,?,?)", rows)
            c.commit()


def start_flusher(interval=1.0):
    def loop():
        while True:
            time.sleep(interval)
            try:
                flush()
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True).start()


def init():
    with closing(_conn()) as c:
        c.execute("PRAGMA journal_mode=WAL")     # lets several containers read/write at once
        c.executescript("""
        CREATE TABLE IF NOT EXISTS uploads(
            id INTEGER PRIMARY KEY, path TEXT, label INTEGER, uploaded_at TEXT, used INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY, ts REAL, label TEXT, confidence REAL, latency_ms REAL);
        CREATE TABLE IF NOT EXISTS retrains(
            id INTEGER PRIMARY KEY, ts TEXT, n_images INTEGER, accepted INTEGER, report TEXT);
        """)
        c.commit()


def add_uploads(files):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with closing(_conn()) as c:
        c.executemany("INSERT INTO uploads(path,label,uploaded_at) VALUES (?,?,?)",
                      [(p, l, now) for p, l in files])
        c.commit()


def pending_uploads():
    with closing(_conn()) as c:
        return [(r["path"], r["label"], r["id"]) for r in
                c.execute("SELECT id,path,label FROM uploads WHERE used=0")]


def mark_used(ids):
    with closing(_conn()) as c:
        c.executemany("UPDATE uploads SET used=1 WHERE id=?", [(i,) for i in ids])
        c.commit()


def upload_counts():
    with closing(_conn()) as c:
        rows = c.execute("SELECT label, used, COUNT(*) n FROM uploads GROUP BY label, used").fetchall()
    out = {"pending": 0, "used": 0, "by_label": {0: 0, 1: 0}}
    for r in rows:
        out["used" if r["used"] else "pending"] += r["n"]
        out["by_label"][r["label"]] += r["n"]
    return out


def log_prediction(label, confidence, latency_ms):
    with _buf_lock:
        _buf.append((time.time(), label, confidence, latency_ms))


def prediction_stats():
    flush()
    with closing(_conn()) as c:
        rows = c.execute("SELECT ts,label,confidence,latency_ms FROM predictions ORDER BY id DESC LIMIT 5000").fetchall()
    return [dict(r) for r in rows]


def log_retrain(n_images, accepted, report):
    with closing(_conn()) as c:
        c.execute("INSERT INTO retrains(ts,n_images,accepted,report) VALUES (?,?,?,?)",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), n_images, int(accepted), json.dumps(report)))
        c.commit()


def retrain_history(limit=10):
    with closing(_conn()) as c:
        rows = c.execute("SELECT * FROM retrains ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "report": json.loads(r["report"])} for r in rows]
