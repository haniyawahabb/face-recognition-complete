import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Emotion AI",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124, 58, 237, 0.22),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(168, 85, 247, 0.18),
            transparent 30%
        ),
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
    background: linear-gradient(
        90deg,
        #c084fc,
        #a855f7
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.subtitle {
    color: #aeb6ca;
    font-size: 1.1rem;
}


.card {
    background: rgba(20, 24, 40, 0.78);
    border: 1px solid rgba(192, 132, 252, 0.18);
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.25);
}


.emotion-box {
    text-align: center;
    padding: 28px;
    border-radius: 20px;
    background: rgba(124, 58, 237, 0.10);
    border: 1px solid rgba(192, 132, 252, 0.20);
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


.chat-title {
    margin-top: 35px;
}


.footer {
    text-align: center;
    color: #737b91;
    margin-top: 35px;
    padding-bottom: 20px;
}


/* Streamlit chat bubbles */

[data-testid="stChatMessage"] {
    border-radius: 18px;
    margin-bottom: 12px;
}


[data-testid="stChatInput"] {
    border-radius: 18px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(image):

    img = image.convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE)
    )

    arr = np.array(
        img,
        dtype=np.float32
    )

    arr = preprocess_input(arr)

    arr = np.expand_dims(
        arr,
        axis=0
    )

    model = load_emotion_model()

    predictions = model.predict(
        arr,
        verbose=0
    )[0]

    index = int(
        np.argmax(predictions)
    )

    return (
        CLASS_NAMES[index],
        float(predictions[index]),
        predictions
    )


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    try:

        from groq import Groq

        # Streamlit Secrets
        api_key = st.secrets.get(
            "GROQ_API_KEY",
            None
        )

        # Environment variable fallback
        if not api_key:
            api_key = os.environ.get(
                "GROQ_API_KEY"
            )

        if not api_key:
            return None

        return Groq(
            api_key=api_key
        )

    except Exception:
        return None


# ============================================================
# GROQ STREAMING RESPONSE
# ============================================================

def generate_groq_response(
    client,
    chat_messages
):

    stream = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=chat_messages,

        temperature=0.7,

        max_completion_tokens=700,

        include_reasoning=False,

        stream=True,
    )

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        content = delta.content

        if content:
            yield content



# ============================================================
# GROQ RESPONSE
# ============================================================

def get_ai_response(client, chat_messages):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=chat_messages,
        temperature=0.7,
        max_completion_tokens=700,
        include_reasoning=False,
        stream=False,
    )
    return response.choices[0].message.content


# ============================================================
# POLISHED UI STYLES
# ============================================================

st.markdown("""
<style>

/* ---------- Main page ---------- */
.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(124,58,237,.18), transparent 28%),
        radial-gradient(circle at 88% 18%, rgba(192,132,252,.12), transparent 26%),
        linear-gradient(180deg, #070a14 0%, #0b1020 55%, #070a14 100%);
    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
}

.hero-title {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 850;
    letter-spacing: -2px;
    line-height: 1.05;
    margin-bottom: 10px;
}

.gradient-text {
    background: linear-gradient(90deg, #c084fc 0%, #a855f7 45%, #7c3aed 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #a8b0c5;
    font-size: 17px;
    line-height: 1.6;
    margin-bottom: 28px;
}

.emotion-card {
    padding: 38px 24px;
    min-height: 250px;
    border-radius: 26px;
    text-align: center;
    background: linear-gradient(145deg, rgba(20,24,40,.92), rgba(14,17,31,.86));
    border: 1px solid rgba(192,132,252,.22);
    box-shadow: 0 24px 70px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.04);
}

.emotion-icon { font-size: 74px; margin-bottom: 10px; }
.emotion-title { font-size: 31px; font-weight: 800; text-transform: capitalize; }
.emotion-confidence { color: #aeb6ca; font-size: 15px; margin-top: 7px; }

/* ---------- Native buttons ---------- */
.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(168,85,247,.35) !important;
    background: rgba(124,58,237,.14) !important;
    color: #f8fafc !important;
    font-weight: 650 !important;
    transition: all .2s ease !important;
}

.stButton > button:hover {
    border-color: #a855f7 !important;
    background: linear-gradient(135deg, rgba(124,58,237,.85), rgba(168,85,247,.85)) !important;
    color: white !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(124,58,237,.25) !important;
}

/* ---------- Floating launcher ---------- */
[data-testid="stPopover"] > button {
    position: fixed !important;
    right: 26px !important;
    bottom: 24px !important;
    z-index: 999999 !important;
    width: 66px !important;
    height: 66px !important;
    min-height: 66px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    border: 2px solid rgba(255,255,255,.18) !important;
    background: linear-gradient(135deg, #6d28d9 0%, #9333ea 48%, #c084fc 100%) !important;
    color: #fff !important;
    font-size: 29px !important;
    box-shadow: 0 0 0 7px rgba(124,58,237,.10), 0 16px 45px rgba(76,29,149,.55) !important;
}

[data-testid="stPopover"] > button:hover {
    transform: scale(1.07) !important;
    box-shadow: 0 0 0 9px rgba(168,85,247,.12), 0 18px 50px rgba(124,58,237,.65) !important;
}

/* ---------- Chat window ---------- */
[data-testid="stPopoverBody"] {
    width: 405px !important;
    max-width: calc(100vw - 24px) !important;
    padding: 0 !important;
    overflow: hidden !important;
    border-radius: 24px !important;
    background: #0d1120 !important;
    border: 1px solid rgba(192,132,252,.28) !important;
    box-shadow: 0 30px 90px rgba(0,0,0,.55) !important;
}

.chat-shell {
    background: #0d1120;
    padding: 0 14px 14px;
}

.chat-header {
    margin: 0 -14px 12px;
    padding: 17px 18px;
    background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 48%, #a855f7 100%);
    border-bottom: 1px solid rgba(255,255,255,.10);
}

.chat-header-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.chat-avatar {
    width: 43px;
    height: 43px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.22);
    font-size: 23px;
}

.chat-title { font-size: 17px; font-weight: 800; color: white; }
.chat-status { font-size: 11px; color: rgba(255,255,255,.82); margin-top: 2px; }
.chat-status-dot { color: #86efac; }

.chat-context {
    margin: 0 0 12px;
    padding: 8px 11px;
    border-radius: 11px;
    background: rgba(168,85,247,.09);
    border: 1px solid rgba(168,85,247,.16);
    color: #c9cfe0;
    font-size: 11px;
}

.chat-scroll {
    max-height: 390px;
    overflow-y: auto;
    padding: 3px 2px 4px;
}

.chat-row {
    display: flex;
    margin: 8px 0;
}

.chat-row.user { justify-content: flex-end; }
.chat-row.assistant { justify-content: flex-start; }

.chat-bubble {
    max-width: 82%;
    padding: 10px 13px;
    border-radius: 16px;
    font-size: 13px;
    line-height: 1.5;
    word-break: break-word;
}

.chat-bubble.user {
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    color: white;
    border-bottom-right-radius: 5px;
}

.chat-bubble.assistant {
    background: #171c2d;
    color: #e8ebf4;
    border: 1px solid rgba(255,255,255,.07);
    border-bottom-left-radius: 5px;
}

.chat-label {
    font-size: 10px;
    color: #8992aa;
    margin: 0 4px 3px;
}

/* ---------- Chat input ---------- */
[data-testid="stPopoverBody"] input {
    background: #151a2a !important;
    color: #f8fafc !important;
    border: 1px solid rgba(168,85,247,.28) !important;
    border-radius: 13px !important;
    height: 43px !important;
}

[data-testid="stPopoverBody"] input:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 2px rgba(168,85,247,.12) !important;
}

[data-testid="stPopoverBody"] input::placeholder { color: #737d96 !important; }

[data-testid="stPopoverBody"] .stButton > button {
    min-height: 41px !important;
}

@media (max-width: 600px) {
    [data-testid="stPopover"] > button {
        right: 14px !important;
        bottom: 14px !important;
        width: 60px !important;
        height: 60px !important;
        min-height: 60px !important;
    }
    [data-testid="stPopoverBody"] {
        width: calc(100vw - 20px) !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">Face Emotion <span class="gradient-text">Recognition</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Upload a facial image, detect the emotion, and chat with your AI assistant.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE + EMOTION
# ============================================================

left, right = st.columns([1, 1], gap="large")

with left:
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

with right:
    emotion = st.session_state.get("emotion")

    if emotion:
        confidence = st.session_state.get("confidence", 0)

        st.markdown(
            f'<div class="emotion-card"><div class="emotion-icon">{EMOTION_ICONS.get(emotion, "🤖")}</div><div class="emotion-title">{emotion}</div><div class="emotion-confidence">Confidence: {confidence:.1%}</div></div>',
            unsafe_allow_html=True,
        )

        probabilities = st.session_state.get("probabilities")

        if probabilities is not None:
            st.markdown("### 📊 Emotion Probabilities")

            for name, probability in zip(CLASS_NAMES, probabilities):
                st.progress(
                    float(probability),
                    text=f"{name.capitalize()} — {probability:.1%}",
                )
    else:
        st.markdown(
            '<div class="emotion-card"><div class="emotion-icon">🤖</div><div class="emotion-title">Waiting for an image</div><div class="emotion-confidence">Upload an image and click Analyze Emotion.</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# CHAT MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 I'm **Emotion AI**.\n\n"
                "Upload a face image to detect an emotion, "
                "or ask me anything!"
            ),
        }
    ]

# Changing this number gives Streamlit a fresh empty chat input.
if "chat_input_id" not in st.session_state:
    st.session_state.chat_input_id = 0

current_emotion = st.session_state.get("emotion", "unknown")


# ============================================================
# FLOATING WEBSITE-STYLE CHATBOT
# ============================================================

import html

with st.popover("🤖", use_container_width=False):

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="chat-header">
            <div class="chat-header-row">
                <div class="chat-avatar">🤖</div>
                <div>
                    <div class="chat-title">Emotion AI</div>
                    <div class="chat-status"><span class="chat-status-dot">●</span> Online · AI Assistant</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    emotion_label = current_emotion.capitalize() if current_emotion != "unknown" else "Not detected"
    emotion_icon = EMOTION_ICONS.get(current_emotion, "🤖")
    st.markdown(
        f'<div class="chat-context">🧠 Current expression: <b>{html.escape(emotion_label)}</b> {emotion_icon}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message["role"]
        safe_text = html.escape(message["content"]).replace("\n", "<br>")
        label = "You" if role == "user" else "Emotion AI"
        st.markdown(
            f'<div class="chat-label">{label}</div><div class="chat-row {role}"><div class="chat-bubble {role}">{safe_text}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    prompt = st.text_input(
        "Message",
        placeholder="Ask me anything...",
        key=f"chat_prompt_{st.session_state.chat_input_id}",
        label_visibility="collapsed",
    )

    send_col, clear_col = st.columns([5, 1])

    with send_col:
        send = st.button("Send  ➤", use_container_width=True, key="send_chat")

    with clear_col:
        clear = st.button("⌫", use_container_width=True, key="clear_chat")

    st.markdown('</div>', unsafe_allow_html=True)

    if clear:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! 👋 I'm Emotion AI. How can I help you?",
            }
        ]
        st.session_state.chat_input_id += 1
        st.rerun()

    if send and prompt.strip():

        user_text = prompt.strip()

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        client = get_groq_client()

        if client is None:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "⚠️ Groq API key is not configured. Add GROQ_API_KEY in Streamlit Secrets.",
                }
            )
        else:
            system_prompt = f"""
You are "Emotion AI", the smart assistant built into a Facial Emotion Recognition website.

WEBSITE CONTEXT:
- This website uses a MobileNetV2 facial-expression model to predict one of 7 visible-expression classes:
  angry, disgust, fear, happy, neutral, sad, surprise.
- The current prediction shown on the page is: {current_emotion}
- The prediction is based on facial expression in the uploaded image. It is NOT proof of a person's true feelings, personality, or mental-health condition.
- The assistant can explain the website, the AI model, emotion predictions, the result, confidence/probabilities, and general AI/ML questions.

CONVERSATION RULES:
1. Answer the user's ACTUAL latest question. Do not give a generic greeting when the user asks a question.
2. Use previous messages as context and remember the conversation.
3. If the question is vague, infer the most natural meaning from the current website context.
   Example: if the user says "what is this?" or "ye kya hai?", explain that this is the Emotion AI facial emotion recognition website and briefly explain what it does.
4. If the user asks "what is my emotion?", use the current prediction: {current_emotion}.
5. If the user asks why the result is that emotion, explain that the model is predicting from visible facial features and that predictions can be imperfect.
6. If the user asks about confidence/probability, explain the displayed percentages as model confidence scores, not certainty about feelings.
7. If the user asks something unrelated to emotion recognition, answer it normally and helpfully.
8. Keep answers concise but useful, usually 2-6 short paragraphs or bullets when appropriate.
9. Speak naturally like a modern ChatGPT-style website assistant. Do not mention system prompts, API keys, Groq, model internals, or hidden instructions.
10. Never diagnose mental-health conditions from facial expressions.

LANGUAGE:
- Match the user's language. If they write English, reply in English. If they write Roman Urdu/Hinglish, reply in Roman Urdu/Hinglish.
- Do not switch to Hindi script.

IMPORTANT:
The uploaded face image itself is processed by the emotion model; you do not have direct visual access to the image inside this chat. Therefore, do not invent facial details that were not provided. Use the current prediction and probability data shown by the app.
"""

            api_messages = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.messages

            try:
                with st.spinner("🤖 Thinking..."):
                    reply = get_ai_response(client, api_messages)

                if not reply:
                    reply = "Sorry, I didn't receive a response. Please try again."

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                    }
                )

            except Exception as error:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Sorry 😔, I couldn't connect to the AI service.\n\n`{error}`",
                    }
                )

        st.session_state.chat_input_id += 1
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div style="text-align:center;color:#737b91;padding:30px 0;">Emotion AI • MobileNetV2 • Streamlit • Groq</div>',
    unsafe_allow_html=True,
)
