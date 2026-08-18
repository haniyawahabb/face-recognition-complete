FROM python:3.11-slim

# System deps needed by pillow / tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .

# Install lightweight deps first (fast, so a flaky connection doesn't waste
# time redoing these). Increased timeout + retries handle slow/unstable
# internet without failing the whole build on one dropped connection.
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    fastapi==0.115.0 uvicorn[standard]==0.30.6 python-multipart==0.0.9 \
    pillow==10.4.0 numpy==1.26.4

# TensorFlow is the big (~220MB) download most likely to hit a timeout on a
# slow connection, so it gets its own layer/retry-friendly install step.
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    tensorflow-cpu==2.17.0

COPY app ./app
COPY models ./models

ENV MODEL_PATH=models/emotion_recognition_mobilenetv2.keras
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
