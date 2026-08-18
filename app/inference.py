"""
Model loading + inference logic for the FER-2013 emotion recognition model
(MobileNetV2 transfer-learning model trained in face_recognition.ipynb).
"""

import io
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 96
MODEL_PATH = os.environ.get("MODEL_PATH", "models/emotion_recognition_mobilenetv2.keras")

# Class order matches flow_from_directory's alphabetical sort of the
# fer2013_data/train subfolders created in the notebook.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_model = None


def load_model():
    """Load the Keras model once and cache it in memory."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Train the model using "
                f"the notebook (or train.py) and place the .keras file there, "
                f"or mount it into the container."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Turn raw uploaded image bytes into a model-ready batch of shape (1, 96, 96, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_emotion(image_bytes: bytes) -> dict:
    """Run inference and return the predicted class + per-class probabilities."""
    model = load_model()
    arr = preprocess_image(image_bytes)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))

    return {
        "predicted_emotion": CLASS_NAMES[idx],
        "confidence": float(preds[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
        },
    }
