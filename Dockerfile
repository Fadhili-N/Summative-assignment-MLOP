FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=0
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, the trained model and small sample data. The full dataset is NOT baked in.
COPY src ./src
COPY app ./app
COPY models ./models
COPY data/sample ./data/sample
COPY data/demo_images ./data/demo_images

# Run as a normal user, not root
RUN useradd -m appuser && mkdir -p /app/data/uploads /state/uploads && chown -R appuser /app /state
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# $PORT is set by most cloud hosts; default 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
