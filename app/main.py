import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from google.genai import types

from app.inference import CLASS_NAMES, load_model, predict_emotion


app = FastAPI(
    title="Facial Emotion Recognition API",
    description="MobileNetV2 transfer-learning model trained on FER-2013.",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Gemini AI
# --------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------
# Allowed image types
# --------------------------------------------------

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
    "image/bmp",
}


# --------------------------------------------------
# Chat request model
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = "unknown"


# --------------------------------------------------
# Startup
# --------------------------------------------------

@app.on_event("startup")
def _load_model_on_startup():

    try:
        load_model()
        print("Model loaded successfully.")

    except FileNotFoundError as e:
        print(f"WARNING: {e}")

    if GEMINI_API_KEY:
        print("Gemini AI configured successfully.")

    else:
        print("WARNING: GEMINI_API_KEY is not configured.")


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Facial Emotion Recognition API is running."
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "classes": CLASS_NAMES,
        "chatbot": "configured" if gemini_client else "not configured"
    }


# --------------------------------------------------
# Emotion Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Upload a JPEG/PNG/WEBP/BMP image."
            ),
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:

        result = predict_emotion(image_bytes)

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}"
        )

    return result


# --------------------------------------------------
# AI CHATBOT
# --------------------------------------------------

@app.post("/chat")
async def chat(request: ChatRequest):

    # Check Gemini configuration
    if gemini_client is None:

        raise HTTPException(
            status_code=503,
            detail="Gemini API is not configured. Please set GEMINI_API_KEY."
        )


    # Validate message
    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )


    # Emotion detected by our ML model
    emotion = request.emotion or "unknown"


    # System instructions for the chatbot
    system_instruction = f"""
You are Emotion AI, a friendly and intelligent emotional-support
chatbot inside a Facial Emotion Recognition application.

The application uses a machine-learning model to detect facial emotions.

The currently detected facial emotion is:

{emotion}

IMPORTANT RULES:

1. Talk naturally like a friendly AI assistant.
2. Answer the user's actual question.
3. Do NOT give the same generic response every time.
4. If the user asks about their emotion, explain the detected emotion.
5. Remember that facial emotion recognition is only an AI prediction,
   not a guaranteed measurement of someone's true feelings.
6. If the detected emotion is happy, respond positively.
7. If the detected emotion is sad, be supportive and gentle.
8. If the detected emotion is angry, respond calmly.
9. If the detected emotion is fearful, be reassuring.
10. If the detected emotion is surprised, respond naturally.
11. If the detected emotion is disgust, respond naturally and politely.
12. If the detected emotion is neutral, explain that the model detected
    a neutral facial expression.
13. You can give general advice, motivation and conversational support.
14. Keep answers concise and easy to understand.
15. Use emojis occasionally, but don't overuse them.
16. Never claim that the model can diagnose a mental-health condition.
17. If the user asks a normal general question, answer that question
    instead of talking about emotions unnecessarily.
"""


    try:

        response = gemini_client.models.generate_content(

            model="gemini-3.8-flash",

            contents=message,

            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=500,
            ),
        )


        reply = response.text


        if not reply:

            raise Exception("Gemini returned an empty response.")


        return {
            "response": reply,
            "emotion": emotion
        }


    except Exception as e:

        print(f"Gemini chatbot error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Chatbot error: {str(e)}"
        )
