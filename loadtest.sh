#!/usr/bin/env bash
# Flood the API with the SAME load against 1, 2 and 3 containers and save the results in results/.
# Usage (project root, Docker running, Locust installed in .venv):  bash loadtest.sh
# Optional: USERS=50 SPAWN=10 DURATION=60s bash loadtest.sh
set -u
USERS=${USERS:-50}; SPAWN=${SPAWN:-10}; DURATION=${DURATION:-60s}
LOCUST=${LOCUST:-.venv/Scripts/locust.exe}
mkdir -p results

for N in 1 2 3; do
  echo "=== $N container(s) ==="
  docker compose up -d --scale app=$N >/dev/null 2>&1
  # wait until all N app containers report healthy
  for i in $(seq 1 60); do
    healthy=$(docker ps --filter "name=malaria-mlops-app" --filter "health=healthy" -q | wc -l)
    [ "$healthy" -ge "$N" ] && break; sleep 3
  done
  sleep 8   # let the load balancer's DNS pick up all replicas
  for i in $(seq 1 20); do curl -s -o /dev/null -F "file=@data/demo_images/parasitized_1.png" localhost:8080/api/predict; done  # warm-up
  $LOCUST -f locustfile.py --host http://localhost:8080 --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" \
          --csv "results/containers_$N" --only-summary 2>&1 | tail -3
done
echo "done: see results/containers_*_stats.csv"
