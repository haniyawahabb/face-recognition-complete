from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.inference import CLASS_NAMES, load_model, predict_emotion

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="MobileNetV2 transfer-learning model trained on FER-2013.",
    version="1.0.0",
)

# Allow calls from any frontend during dev; tighten this for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}


@app.on_event("startup")
def _load_model_on_startup():
    # Fail fast with a clear log line if the model file is missing,
    # instead of failing on the first request.
    try:
        load_model()
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")


@app.get("/")
def root():
    return {"status": "ok", "message": "Facial Emotion Recognition API is running."}


@app.get("/health")
def health():
    return {"status": "healthy", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG/PNG/WEBP/BMP image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_emotion(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result