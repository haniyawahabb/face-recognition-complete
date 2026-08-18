# Facial Emotion Recognition — FastAPI + Docker

MobileNetV2 transfer-learning model (trained on FER-2013) served through a FastAPI
`/predict` endpoint, containerized with Docker.

> Note: the original notebook (`face_recognition.ipynb`) trains an **emotion**
> classifier (angry, disgust, fear, happy, neutral, sad, surprise) — not a
> person-identity face recognizer. This API wraps that model as-is.

## Project structure

```
.
├── app/
│   ├── main.py         # FastAPI app (/predict, /health)
│   └── inference.py     # model loading + preprocessing
├── models/
│   └── emotion_recognition_mobilenetv2.keras   # <-- put your trained model here
├── train.py             # standalone script version of the notebook
├── face_recognition.ipynb
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## 1. Get a trained model file

The notebook doesn't save a model to this repo — you need to produce
`models/emotion_recognition_mobilenetv2.keras` first, either:

- Run `face_recognition.ipynb` in Colab (as before) and download the saved
  `.keras` file into `models/`, **or**
- Run `python train.py` locally/on a GPU box (installs deps from
  `requirements.txt` + `gdown`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
  since those aren't needed at inference time but are needed for training).

## 2. Run locally (no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## 3. Run with Docker

```bash
docker build -t emotion-recognition-api .
docker run -p 8000:8000 emotion-recognition-api
```

If your model file isn't baked into the image, mount it instead:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/code/models emotion-recognition-api
```

## 4. Call the API

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/face.jpg"
```

Response:

```json
{
  "predicted_emotion": "happy",
  "confidence": 0.87,
  "probabilities": {
    "angry": 0.01,
    "disgust": 0.00,
    "fear": 0.02,
    "happy": 0.87,
    "neutral": 0.05,
    "sad": 0.03,
    "surprise": 0.02
  }
}
```

## Endpoints

| Method | Path       | Description                          |
|--------|-----------|---------------------------------------|
| GET    | `/`       | Health/status message                 |
| GET    | `/health` | Health check + class list             |
| POST   | `/predict`| Upload an image, get emotion + probs  |
