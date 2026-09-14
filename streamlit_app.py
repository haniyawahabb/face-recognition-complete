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
        model="llama-3.3-70b-versatile",

        messages=chat_messages,

        temperature=0.7,

        max_tokens=700,

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
        model="llama-3.3-70b-versatile",
        messages=chat_messages,
        temperature=0.7,
        max_tokens=700,
        stream=False,
    )
    return response.choices[0].message.content


# ============================================================
# CLEAN UI STYLES
# ============================================================

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(124,58,237,.20), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(168,85,247,.16), transparent 30%),
                #080b16;
}
.block-container { max-width: 1150px; padding-top: 2rem; }
.hero-title { font-size: 48px; font-weight: 800; line-height: 1.1; margin-bottom: 8px; }
.gradient-text { color: #c084fc; }
.hero-subtitle { color: #aeb6ca; font-size: 17px; margin-bottom: 30px; }
.emotion-card {
    padding: 32px;
    border-radius: 22px;
    text-align: center;
    background: rgba(20,24,40,.78);
    border: 1px solid rgba(192,132,252,.20);
    box-shadow: 0 15px 50px rgba(0,0,0,.25);
}
.emotion-icon { font-size: 70px; }
.emotion-title { font-size: 30px; font-weight: 700; text-transform: capitalize; }
.emotion-confidence { color: #aeb6ca; font-size: 15px; }

/* Floating chatbot launcher */
[data-testid="stPopover"] > button {
    position: fixed !important;
    right: 28px !important;
    bottom: 22px !important;
    z-index: 999999 !important;
    width: 62px !important;
    height: 62px !important;
    min-height: 62px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    border: 0 !important;
    background: linear-gradient(135deg,#7c3aed,#a855f7) !important;
    color: white !important;
    font-size: 27px !important;
    box-shadow: 0 12px 35px rgba(124,58,237,.45) !important;
}
[data-testid="stPopover"] > button:hover { transform: scale(1.05); }
[data-testid="stPopoverBody"] {
    width: 390px !important;
    max-width: calc(100vw - 30px) !important;
    border-radius: 20px !important;
}
.chat-head {
    background: linear-gradient(135deg,#7c3aed,#a855f7);
    color: white;
    padding: 14px 16px;
    border-radius: 15px;
    margin-bottom: 12px;
}
.chat-head-title { font-size: 18px; font-weight: 700; }
.chat-head-status { font-size: 12px; opacity: .9; }
@media (max-width: 600px) {
    [data-testid="stPopover"] > button {
        right: 15px !important;
        bottom: 15px !important;
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

with st.popover("🤖", use_container_width=False):

    st.markdown(
        '<div class="chat-head"><div class="chat-head-title">🤖 Emotion AI</div><div class="chat-head-status">● Online • AI Assistant</div></div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Detected emotion: {EMOTION_ICONS.get(current_emotion, '🤖')} {current_emotion.capitalize()}"
    )

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"],
            avatar="🤖" if message["role"] == "assistant" else "👤",
        ):
            st.markdown(message["content"])

    prompt = st.text_input(
        "Message",
        placeholder="Type your message...",
        key=f"chat_prompt_{st.session_state.chat_input_id}",
        label_visibility="collapsed",
    )

    send_col, clear_col = st.columns([4, 1])

    with send_col:
        send = st.button(
            "➤ Send",
            use_container_width=True,
            key="send_chat",
        )

    with clear_col:
        clear = st.button(
            "🗑️",
            use_container_width=True,
            key="clear_chat",
        )

    if clear:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! 👋 I'm **Emotion AI**. How can I help you?",
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
                    "content": (
                        "⚠️ **Groq API key is not configured.**\n\n"
                        "Add `GROQ_API_KEY` in Streamlit Secrets."
                    ),
                }
            )
        else:
            system_prompt = f"""
You are Emotion AI, a friendly and intelligent AI assistant inside a Facial Emotion Recognition website.

Current detected facial emotion: {current_emotion}

Have a natural conversation like a normal website AI chatbot.
Answer the user's actual question, remember previous messages,
and be clear, friendly and helpful.

If asked about the detected emotion, explain that it is only an AI
prediction based on facial expression and does not prove the person's
actual feelings. Never diagnose mental-health conditions from facial emotion.

If the user asks something unrelated to emotion recognition, answer normally.
"""

            api_messages = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.messages

            try:
                with st.spinner("🤖 Typing..."):
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
                        "content": (
                            "Sorry 😔, I couldn't connect to the AI service.\n\n"
                            f"`{error}`"
                        ),
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
