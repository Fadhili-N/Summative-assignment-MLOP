"""Load test for the malaria classifier API.

Interactive UI:   locust -f locustfile.py --host http://localhost:8080
Headless example: locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s --csv results/run

Each simulated user mostly uploads a cell image for prediction (the expensive call)
and sometimes checks /api/health, like the web page does.
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGES = [(p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / "data" / "demo_images").glob("*.png"))]


class ClinicUser(HttpUser):
    wait_time = between(0.1, 0.5)      # short think time to create a flood of requests

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post("/api/predict", files={"file": (name, data, "image/png")},
                              name="POST /api/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif r.json().get("label") not in ("Parasitized", "Uninfected"):
                r.failure("unexpected response")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="GET /api/health")
