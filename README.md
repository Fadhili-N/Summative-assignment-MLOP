---
title: Malaria Cell Classifier
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Malaria Cell Classifier: an end-to-end MLOps pipeline

A web app and API that looks at a microscope image of a single blood cell and says whether it is **Parasitized** (infected with malaria) or **Uninfected**. It covers the full pipeline: data acquisition, preprocessing, model training, testing, a prediction API, bulk data upload, **retraining on new data**, monitoring, Docker deployment and a load test.

> Study project. Not a medical device and not for diagnosis.

| | |
|---|---|
| **Video demo (YouTube)** | `TODO: paste link` |
| **Live app** | `TODO: paste Hugging Face Space URL` |
| **GitHub repo** | `TODO: paste link` |

## What you can do in the app

| Page | What it does |
|---|---|
| **Predict** | Upload one cell image, get the label, a confidence score and the response time. |
| **Upload & Retrain** | Upload many images at once (a `.zip` with `Parasitized/` and `Uninfected/` folders, or several images plus a label). They are validated, saved to a database, and used to fine-tune the model with the **Retrain** button. Retraining also starts **automatically** once 50 new images are waiting. |
| **Data insights** | Charts comparing infected and uninfected cells on five measured image features, each with a written interpretation. |
| **Status** | Uptime, model version, test-set metrics, number of predictions served, latency, and a confidence histogram that helps spot drift. |

## Data

[NIH malaria cell images](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html): **27,558** cell images, almost perfectly balanced (13,779 per class), loaded through TensorFlow Datasets. Split used by the notebook: **70% train / 15% validation / 15% test** (the test set is untouched until the final evaluation). Images are **not** stored in this repo; `python -m src.data_acquisition` downloads them into `data/train/` and `data/test/`.

## Model

Transfer learning: **MobileNetV2** pretrained on ImageNet, a small dense head, input 128x128. Four experiments were compared on the validation set (each in its own notebook cell):

1. Baseline, frozen base
2. + dropout 0.3 and L2 regularization
3. + data augmentation (flip, rotation, zoom)
4. + fine-tuning the last 30 layers at a low learning rate (selected as best)

Optimization techniques: pretrained model, regularization (dropout, L2), augmentation, Adam with different learning rates, early stopping, learning-rate reduction on plateau, fine-tuning.

### Results on the held-out test set (best model, `exp4_finetune`)

Positive class = Parasitized (a missed infection is the costly mistake).

| Accuracy | Precision | Recall | F1 | AUC | Loss |
|---|---|---|---|---|---|
| 0.958 | 0.968 | 0.948 | 0.958 | 0.990 | 0.135 |

About 1 in 20 infected cells is missed (recall 94.8%): 173 of the 4,134 test images are wrong.

**Is the confidence score trustworthy?** Partly. Wrong predictions are much less confident on average (0.79 vs 0.97 for correct ones), and 35% of errors fall below 70% confidence compared with only 3.5% of all predictions, so the app flags anything below 70%. But 36% of errors are 90% or more confident, so the model is sometimes confidently wrong. When confidence is at least 90% (about 90% of images) accuracy is 98.3%. Treat confidence as a warning signal, not a guarantee.

`TODO: paste the experiment comparison table from notebook section 6 here.`

## Data insights (what the images tell us)

Measured on 3,000 randomly sampled training images (1,500 per class). Effect size *d* says how far apart the class averages are (about 0.2 small, 0.5 medium, 0.8 large).

| Feature | Parasitized | Uninfected | d | Story |
|---|---|---|---|---|
| Texture / contrast | 17.2 | 15.5 | 0.72 | Strongest cue: a dark parasite against paler cell material adds contrast inside the cell. |
| Brightness | 162.6 | 167.7 | -0.56 | Infected cells are slightly darker (stained parasite). |
| Colour saturation | 0.254 | 0.202 | 0.53 | The stain adds colour. |
| Cell size | 0.721 | 0.733 | -0.21 | Almost the same, so the model cannot cheat on size. |
| Raw image size | 18.2k px | 17.5k px | 0.13 | Almost the same, so resizing loses no hidden label information. |

Every single feature overlaps heavily between classes, which is why a CNN combining many subtle patterns (95.8% accuracy) beats any one measurement. The full text is in `models/insight_stories.json` and in the app's *Data insights* page.

## Retraining

1. **Upload:** images arrive through `POST /api/upload`. Only real images are kept (each is opened and verified), names are regenerated (a zip cannot write outside the upload folder), and each file is recorded in SQLite.
2. **Preprocess:** resize to 128x128 (the scaling to [-1, 1] is inside the model).
3. **Retrain:** the **saved model is the pretrained starting point** and is fine-tuned at a low learning rate, mixed with some original images so it does not forget them.
4. **Safety check:** the new model replaces the live one only if it is not worse (F1 within 0.01) on both (a) a held-out slice of the new data and (b) 200 fixed images from the original test set. The old model is backed up first. Results are shown in the UI and kept in a history table.
5. **Trigger:** manual (button / `POST /api/retrain`) or automatic when 50 new images are waiting (`RETRAIN_THRESHOLD`).

With a tiny learning rate and a few epochs, retraining changes the model gently. Retraining on very different images (for example, a different microscope) is where it visibly matters.

## Load test (Locust, flood of requests)

Same load each time: 50 simulated users, 60 seconds, mostly `POST /api/predict` with a cell image. Each container is limited to 1 CPU core and runs behind an nginx load balancer. Everything ran on one 8-core laptop (Locust, nginx and all containers share it), 0 failed requests in every run.

| Containers | Requests | Requests/sec | Avg latency | Median | 95th pct | 99th pct |
|---|---|---|---|---|---|---|
| 1 | 1,814 | 31.6 | 1,170 ms | 1,100 ms | 1,600 ms | 2,100 ms |
| 2 | 3,138 | 54.6 | 542 ms | 480 ms | 1,300 ms | 1,500 ms |
| 3 | 4,561 | 78.0 | 277 ms | 200 ms | 710 ms | 930 ms |

How to read it: a single idle request takes about 26 ms; the latencies above are high because 50 users flood a 1-core container and requests queue. More containers cut latency and raise throughput (2 containers gave 1.7x the requests per second, 3 gave 2.5x, not a perfect 2x and 3x, because they share one machine's CPU). Each configuration was run once and the first seconds of ramp-up are included. Raw files are in `results/`.

Lesson learned while measuring: TensorFlow started 8 threads inside a 1-CPU container, which made each request 2.5x slower. Setting `TF_NUM_INTRAOP_THREADS=1` fixed it.

## Run it yourself

**Option A: Docker (recommended)**
```bash
docker compose up --build -d --scale app=3     # 3 copies behind a load balancer
# open http://localhost:8080
```
Change `--scale app=3` to run 1, 2 or more containers.

**Option B: locally with Python 3.13**
```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --port 8000      # open http://localhost:8000
```

**Get the dataset** (only needed for insights / full retraining checks, about 1 GB of downloads):
```bash
pip install -r requirements-dev.txt
python -m src.data_acquisition        # writes data/train and data/test
python -m src.insights                # recomputes models/insights.json
```

**Run the load test**
```bash
pip install -r requirements-dev.txt
bash loadtest.sh                      # 1, 2 and 3 containers, results in results/
# or interactively:  locust -f locustfile.py --host http://localhost:8080
```

**Retrain the notebook model:** open `notebook/malaria_mlops.ipynb` in Google Colab (GPU runtime) and run all cells. It downloads the data itself and saves `models/malaria_model.h5`.

## API

Interactive docs (Swagger UI) at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /api/predict` | one image file, returns label, confidence, latency |
| `POST /api/upload` | `.zip` or several images (+ `label`), saved for retraining |
| `POST /api/retrain` | start retraining on the uploaded images |
| `GET /api/retrain/status` | running / last result / history |
| `GET /api/health` | uptime, model version, test metrics |
| `GET /api/monitor` | prediction counts, latency, confidence histogram |
| `GET /api/insights` | data behind the insight charts |

**Security note:** anyone who can reach the app can upload and retrain unless you set an `ADMIN_TOKEN` environment variable. When it is set, upload and retrain require the `X-Admin-Token` header (there is a token box in the UI). Set it on any public deployment.

## Deploying to Hugging Face Spaces

1. Create a new Space, choose the **Docker** SDK, and note its git URL.
2. In the Space settings, add a secret `ADMIN_TOKEN` (your choice of password).
3. Push this repo to the Space's git remote. The front matter at the top of this README configures it (port 8000).
4. Wait for the build, then open the Space URL.

The free tier's disk is temporary: uploaded images, the database and retrained models are lost when the Space restarts. The original model is always restored from the repo.

## Project structure

```
malaria-mlops/
├── README.md
├── notebook/malaria_mlops.ipynb   data -> preprocessing -> 4 experiments -> evaluation -> save model
├── src/
│   ├── data_acquisition.py        download + write data/train, data/test
│   ├── preprocessing.py           image loading, validation, safe zip extraction
│   ├── model.py                   evaluation + retraining (fine-tunes the saved model)
│   ├── prediction.py              single-image prediction
│   └── insights.py                feature statistics for the charts
├── app/                           FastAPI backend (main.py, db.py) + web UI (static/index.html)
├── data/train, data/test          (downloaded, not committed)  data/sample, data/demo_images (small, committed)
├── models/                        malaria_model.h5, model_info.json, insights.json
├── results/                       Locust CSV results
├── Dockerfile, docker-compose.yml, nginx.conf, locustfile.py, loadtest.sh
└── requirements.txt, requirements-dev.txt
```

## Limitations

- One public dataset from one source; performance on slides from other labs or microscopes is unknown and likely lower.
- The model sees single cropped cells, not whole slides.
- Recall is 94.8%: some infected cells are missed. Not suitable for clinical use.
- The load test ran on one laptop, one run per configuration.
