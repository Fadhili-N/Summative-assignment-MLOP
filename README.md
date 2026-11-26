# Malaria Cell Classifier

This is my MLOps summative project. It's a web app and an API that looks at a picture of a single blood cell and tells you if it's **Parasitized** (infected with malaria) or **Uninfected**. I built the whole pipeline around the model: getting the data, preprocessing, training, testing, an API, uploading new data, retraining, monitoring, Docker, a load test, and deploying it to Azure.

This is a school project, not a medical tool. Please don't use it to diagnose anyone.

| | |
|---|---|
| **Video demo (YouTube)** | `TODO: paste link` |
| **Live app** | http://172.189.158.124:8000 (Azure, France Central, plain `http`, API docs at `/docs`) |
| **GitHub repo** | https://github.com/Fadhili-N/Summative-assignment-MLOP |

## What the app does

| Page | What it does |
|---|---|
| **Predict** | You upload one cell image and it gives you the label, a confidence score and how long it took. |
| **Upload & Retrain** | You upload lots of images at once (a `.zip` with `Parasitized/` and `Uninfected/` folders, or several images plus a label). They get checked, saved to a database, and used to fine-tune the model when you press **Retrain**. Retraining also starts by itself once 50 new images are waiting. |
| **Data insights** | Charts comparing infected and uninfected cells on five things I measured from the images, with my explanation of each. |
| **Status** | Uptime, model version, test metrics, how many predictions it served, latency, and a confidence histogram (useful for spotting if the incoming data starts to look different). |

## The data

I used the [NIH malaria cell images](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html): **27,558** images, perfectly balanced (13,779 per class), loaded through TensorFlow Datasets. In the notebook I split it **70% train, 15% validation, 15% test**, and I only used the test set at the very end. The images are not in this repo. `python -m src.data_acquisition` downloads them into `data/train/` and `data/test/`.

## The model

I used transfer learning: **MobileNetV2** pretrained on ImageNet, with a small dense layer on top, and images resized to 128x128. I compared four experiments on the validation set, each in its own notebook cell:

1. Baseline, base frozen
2. Plus dropout 0.3 and L2 regularization
3. Plus data augmentation (flip, rotation, zoom)
4. Plus fine-tuning the last 30 layers with a low learning rate (this one won)

Things I used to improve the model: a pretrained model, dropout, L2, augmentation, the Adam optimizer with different learning rates, early stopping, reducing the learning rate when it stops improving, and fine-tuning. Early stopping was switched on, but every run still used all of its epochs (10 for experiments 1 to 3, 12 for experiment 4), so training for longer might change the comparison below.

### Experiment comparison (validation set)

| Experiment | Accuracy | Precision | Recall | F1 | AUC | Loss |
|---|---|---|---|---|---|---|
| exp1_baseline (frozen base) | 0.9451 | 0.9471 | 0.9420 | 0.9445 | 0.9859 | 0.1484 |
| exp2_dropout_l2 | 0.9405 | 0.9493 | 0.9298 | 0.9394 | 0.9838 | 0.1709 |
| exp3_augmentation | 0.9231 | 0.9446 | 0.8977 | 0.9205 | 0.9760 | 0.2128 |
| **exp4_finetune (selected)** | **0.9644** | **0.9685** | **0.9596** | **0.9640** | **0.9932** | **0.1137** |

What I noticed:
- **Fine-tuning helped the most.** Unfreezing the last 30 layers took F1 from 0.944 to 0.964. Letting the pretrained layers adjust to cell images mattered more than anything else I tried.
- **Dropout/L2 and augmentation did not help the frozen model.** F1 went down to 0.939 and 0.921, and augmentation hurt recall the most (0.898). My guess is that the frozen model wasn't overfitting, so the extra regularization just made it harder to learn, but I didn't test that, so it's only a guess. Experiment 4 uses both of them, and I didn't run fine-tuning without them, so I can't say which part of experiment 4 deserves the credit.
- **Validation F1 of the selected model is 0.964 but test F1 is 0.958.** The small drop makes sense, because I picked the model using the validation set. The test numbers below are the honest ones.

A small note about the model file: the notebook saves the full model, and the `models/malaria_model.h5` in this repo is the same model with the optimizer state removed (22 MB down to 9.7 MB). I checked that the predictions are identical on 300 test images, and retraining still works from it.

### Test set results (best model, exp4_finetune)

I treat Parasitized as the positive class, because missing an infected cell is the worst mistake.

| Accuracy | Precision | Recall | F1 | AUC | Loss |
|---|---|---|---|---|---|
| 0.958 | 0.968 | 0.948 | 0.958 | 0.990 | 0.135 |

About 1 in 20 infected cells gets missed (recall 94.8%). The model got 173 of the 4,134 test images wrong.

**Can I trust the confidence score?** Only partly. When the model is wrong it's usually less sure (average confidence 0.79, compared to 0.97 when it's right), and 35% of its mistakes are under 70% confidence, compared to only 3.5% of all predictions. That's why the app shows a warning under 70%. But 36% of its mistakes are 90% confident or higher, so it's sometimes confidently wrong. When confidence is 90% or higher (about 90% of images), accuracy is 98.3%. So I treat confidence as a hint, not a guarantee.

## What the images tell us

I measured five simple things on 3,000 random training images (1,500 per class). The "d" column is the effect size, which tells you how far apart the two class averages are (around 0.2 is small, 0.5 medium, 0.8 large).

| Feature | Parasitized | Uninfected | d | What I think it means |
|---|---|---|---|---|
| Texture / contrast | 17.2 | 15.5 | 0.72 | The strongest one. A dark parasite against a paler cell adds contrast inside the cell. |
| Brightness | 162.6 | 167.7 | -0.56 | Infected cells are slightly darker, probably because of the stain on the parasite. |
| Colour saturation | 0.254 | 0.202 | 0.53 | The stain adds colour. |
| Cell size | 0.721 | 0.733 | -0.21 | Almost the same, so the model can't cheat by judging size. |
| Raw image size | 18.2k px | 17.5k px | 0.13 | Almost the same, so resizing doesn't throw away hidden information about the label. |

Every one of these overlaps a lot between the two classes, so no single measurement could do the job on its own. That's why a CNN that combines lots of small patterns (95.8% accuracy) works better. My full explanations are in `models/insight_stories.json` and on the app's Data insights page.

## How retraining works

1. **Upload.** Images come in through `POST /api/upload`. Each file is opened and checked to be a real image, the file name is replaced with a generated one (so a malicious zip can't write files outside the upload folder), and each image is recorded in an SQLite database.
2. **Preprocess.** Images are resized to 128x128. The pixel scaling happens inside the model.
3. **Retrain.** My saved model is the pretrained starting point. It gets fine-tuned with a low learning rate, on the new images mixed with some of the original ones so it doesn't forget them.
4. **Safety check.** The new model only replaces the live one if it's not worse (F1 within 0.01) on both a held-out slice of the new data and 200 fixed images from the original test set. The old model is backed up first. The results show up in the UI and in a history table.
5. **Trigger.** You can press the button (`POST /api/retrain`), or it starts automatically when 50 new images are waiting (`RETRAIN_THRESHOLD`).

With a small learning rate and a few epochs, retraining only changes the model a little. It would matter more with very different images, for example from a different microscope.

## Flood test with Locust

I wanted to see how the app copes with lots of requests. Each run used 50 simulated users for 60 seconds, mostly sending cell images to `POST /api/predict`. I ran the app in Docker behind an nginx load balancer, with each container limited to 1 CPU core, and I repeated the same test with 1, 2 and 3 containers. Everything ran on my one 8-core laptop (Locust, nginx and the containers share it). There were 0 failed requests in every run.

| Containers | Requests | Requests/sec | Avg latency | Median | 95th pct | 99th pct |
|---|---|---|---|---|---|---|
| 1 | 1,814 | 31.6 | 1,170 ms | 1,100 ms | 1,600 ms | 2,100 ms |
| 2 | 3,138 | 54.6 | 542 ms | 480 ms | 1,300 ms | 1,500 ms |
| 3 | 4,561 | 78.0 | 277 ms | 200 ms | 710 ms | 930 ms |

How I read this: one request on its own takes about 26 ms. The latencies are high in the table only because 50 users hit a 1-core container at the same time, so requests have to queue. More containers means more requests per second and lower latency. Two containers gave 1.7 times the requests per second and three gave 2.5 times, not a clean 2 and 3 times, which I think is because they all share my laptop's CPU. I ran each setup once, and the first few seconds of ramp-up are included. The raw files are in `results/`.

Something I learned while doing this: my first run was much slower than expected. TensorFlow was starting 8 threads inside a container that only had 1 CPU, and each request took about 65 ms instead of about 26 ms. Setting `TF_NUM_INTRAOP_THREADS=1` fixed it.

## How to run it

**With Docker (the easiest way)**
```bash
docker compose up --build -d --scale app=3     # 3 copies behind a load balancer
# open http://localhost:8080
```
Change `--scale app=3` to run 1, 2 or more containers.

**Locally with Python 3.13**
```bash
python -m venv .venv
.venv\Scripts\activate                # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --port 8000      # open http://localhost:8000
```

**Getting the dataset** (only needed for the insights and for the full retraining check; it's about 1 GB of downloads)
```bash
pip install -r requirements-dev.txt
python -m src.data_acquisition        # writes data/train and data/test
python -m src.insights                # recomputes models/insights.json
```

**Running the load test**
```bash
pip install -r requirements-dev.txt
bash loadtest.sh                      # 1, 2 and 3 containers, results go to results/
# or interactively:  locust -f locustfile.py --host http://localhost:8080
```

**Retraining the notebook model:** open `notebook/malaria_mlops.ipynb` in Google Colab with a GPU, and run all the cells. It downloads the data by itself and saves `models/malaria_model.h5`.

## API

The interactive docs (Swagger UI) are at `/docs`.

| Endpoint | What it does |
|---|---|
| `POST /api/predict` | one image in, label + confidence + latency out |
| `POST /api/upload` | a `.zip` or several images (+ a `label`), saved for retraining |
| `POST /api/retrain` | starts retraining on the uploaded images |
| `GET /api/retrain/status` | running or not, last result, history |
| `GET /api/health` | uptime, model version, test metrics |
| `GET /api/monitor` | prediction counts, latency, confidence histogram |
| `GET /api/insights` | the data behind the insight charts |

**Security note:** if you don't set an `ADMIN_TOKEN` environment variable, anyone who can reach the app can upload data and retrain it. When it's set, upload and retrain need the `X-Admin-Token` header (the UI has a token box). I set it on the Azure deployment.

## Cloud deployment (Azure)

I'm running the same Docker image on Azure Container Instances, using my Azure for Students subscription:

1. Build the image (`docker compose build`) and push it to Docker Hub: `docker tag malaria-mlops-app YOURUSER/malaria:v1`, then `docker push YOURUSER/malaria:v1`.
2. In the Azure Portal, create a **Container Instance**: Linux, image `YOURUSER/malaria:v1` (public), 2 vCPUs and 4 GiB memory, public port **8000/TCP**.
3. Add these environment variables: `ADMIN_TOKEN` (marked secure, it's the password for upload and retrain), `TF_NUM_INTRAOP_THREADS=2` and `TF_NUM_INTEROP_THREADS=1`.
4. Open `http://<public IP>:8000`.

Things to know:
- Student subscriptions can only deploy to a few regions. France Central worked for me. East US and Sweden Central were blocked by a policy.
- The container's disk is temporary. If it restarts, uploaded images, the database and retrained models are lost, and the original model comes back from the image.
- The address is a plain-`http` IP address and it can change if the container is deleted and recreated.
- A running container uses my Azure credit, so I stop it when I'm not using it. If the link doesn't respond, it may be stopped.
- I checked it from outside Azure on 2026-11-26: `/api/health` returned the model metrics, and `/api/predict` classified a parasitized and an uninfected demo image correctly.

## Project structure

```
malaria-mlops/
├── README.md
├── notebook/malaria_mlops.ipynb   data -> preprocessing -> 4 experiments -> evaluation -> save model
├── src/
│   ├── data_acquisition.py        downloads the data into data/train and data/test
│   ├── preprocessing.py           image loading, validation, safe zip extraction
│   ├── model.py                   evaluation + retraining (fine-tunes the saved model)
│   ├── prediction.py              single-image prediction
│   └── insights.py                the feature statistics for the charts
├── app/                           FastAPI backend (main.py, db.py) + web UI (static/index.html)
├── data/train, data/test          (downloaded, not committed)
│   data/sample, data/demo_images  (small, committed)
├── models/                        malaria_model.h5, model_info.json, insights.json
├── results/                       Locust CSV results
├── Dockerfile, docker-compose.yml, nginx.conf, locustfile.py, loadtest.sh
└── requirements.txt, requirements-dev.txt
```

## Limitations

- I only used one public dataset from one source, so I don't know how well it works on slides from other labs or microscopes. I'd expect it to do worse.
- The model only sees single cropped cells, not whole slides.
- Recall is 94.8%, so some infected cells are missed. It's not suitable for real diagnosis.
- I ran the load test on one laptop, once per setup.
