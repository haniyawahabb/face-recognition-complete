import io
import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Emotion AI",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 96
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "emotion_recognition_mobilenetv2.keras",
)

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_ICONS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124,58,237,.22), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,.18), transparent 30%),
        #080b16;
    color: #f5f7ff;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

.hero {
    padding: 30px 10px 20px;
}

.hero h1 {
    font-size: 3.3rem;
    line-height: 1.05;
    margin-bottom: 12px;
}

.gradient {
    background: linear-gradient(90deg, #c084fc, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #aeb6ca;
    font-size: 1.1rem;
}

.card {
    background: rgba(20,24,40,.78);
    border: 1px solid rgba(192,132,252,.18);
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 15px 50px rgba(0,0,0,.25);
}

.emotion-box {
    text-align: center;
    padding: 28px;
    border-radius: 20px;
    background: rgba(124,58,237,.10);
    border: 1px solid rgba(192,132,252,.20);
}

.emotion-icon {
    font-size: 5rem;
}

.emotion-name {
    font-size: 2rem;
    font-weight: 700;
    text-transform: capitalize;
}

.small {
    color: #aeb6ca;
}

.chat-box {
    background: rgba(15,18,31,.85);
    border-radius: 20px;
    border: 1px solid rgba(192,132,252,.18);
    padding: 18px;
}

.bot {
    background: rgba(124,58,237,.15);
    border-radius: 16px;
    padding: 12px 16px;
    margin: 8px 0;
}

.user {
    background: rgba(255,255,255,.07);
    border-radius: 16px;
    padding: 12px 16px;
    margin: 8px 0;
}

.footer {
    text-align:center;
    color:#737b91;
    margin-top:35px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_emotion(image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    model = load_emotion_model()
    predictions = model.predict(arr, verbose=0)[0]
    index = int(np.argmax(predictions))

    return CLASS_NAMES[index], float(predictions[index]), predictions

# -----------------------------
# Gemini
# -----------------------------
@st.cache_resource
def get_gemini_client():
    try:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            return None
        return genai.Client(api_key=key)
    except Exception:
        return None

def ask_ai(message, emotion):
    client = get_gemini_client()

    if client is None:
        return (
            "I'm connected to the Emotion AI app, but the Gemini API key "
            "has not been configured yet. Please add GEMINI_API_KEY in "
            "Streamlit Secrets."
        )

    instruction = f"""
You are Emotion AI, a friendly chatbot inside a facial emotion
recognition application.

The machine-learning model currently detected this facial expression:
{emotion}

The detection is only an AI prediction and does not prove exactly
how the person feels.

Answer the user's actual question naturally.
If they ask about their emotion, explain the detected expression.
If they want advice, give practical and supportive general advice.
If they ask a normal question, answer it normally.
Do not repeat generic replies.
Keep responses concise and friendly.
Use occasional emojis.
Never diagnose a mental-health condition from facial emotion.

User message:
{message}
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=instruction,
        )
        return response.text or "I couldn't generate a response right now."
    except Exception as e:
        return f"AI connection error: {e}"

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <div style="color:#c084fc;font-weight:700;letter-spacing:2px;">
        ✦ AI POWERED
    </div>
    <h1>Face Emotion <span class="gradient">Recognition</span></h1>
    <div class="subtitle">
        Upload a facial image, detect the emotion, and chat with your AI assistant.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main columns
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Analyze an Image")

    uploaded = st.file_uploader(
        "Choose a clear face image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Selected image", use_container_width=True)

        if st.button("🧠 Analyze Emotion", use_container_width=True):
            with st.spinner("Analyzing facial expression..."):
                emotion, confidence, probabilities = predict_emotion(image)

            st.session_state["emotion"] = emotion
            st.session_state["confidence"] = confidence
            st.session_state["probabilities"] = probabilities

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    emotion = st.session_state.get("emotion")

    if emotion:
        confidence = st.session_state.get("confidence", 0)

        st.markdown(
            f"""
            <div class="card emotion-box">
                <div class="small">Detected Emotion</div>
                <div class="emotion-icon">{EMOTION_ICONS.get(emotion, "🤖")}</div>
                <div class="emotion-name">{emotion}</div>
                <div class="small">Confidence: {confidence:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probabilities = st.session_state.get("probabilities")
        if probabilities is not None:
            st.markdown("### 📊 Emotion Probabilities")
            for name, probability in zip(CLASS_NAMES, probabilities):
                st.progress(float(probability), text=f"{name.capitalize()} — {probability:.1%}")
    else:
        st.markdown("""
        <div class="card emotion-box">
            <div class="emotion-icon">🤖</div>
            <div class="emotion-name">Waiting for an image</div>
            <div class="small">Upload an image and click Analyze Emotion.</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Chatbot
# -----------------------------
st.markdown("## 💬 Emotion AI Chatbot")

current_emotion = st.session_state.get("emotion", "unknown")

st.markdown(
    f"""
    <div class="card">
        <div class="small">Current detected emotion</div>
        <strong>{EMOTION_ICONS.get(current_emotion, "🤖")} {current_emotion.capitalize()}</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 I'm your Emotion AI assistant. Upload an image first, or just start chatting with me."
        }
    ]

for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.markdown(
            f'<div class="bot">🤖 <strong>Emotion AI</strong><br>{message["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="user">👤 <strong>You</strong><br>{message["content"]}</div>',
            unsafe_allow_html=True,
        )

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Emotion AI is typing..."):
        reply = ask_ai(prompt, current_emotion)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

st.markdown(
    '<div class="footer">Face Emotion Recognition • MobileNetV2 + Streamlit + AI</div>',
    unsafe_allow_html=True,
)
