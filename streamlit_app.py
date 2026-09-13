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
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124,58,237,.22),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(168,85,247,.18),
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
    text-align: center;
    color: #737b91;
    margin-top: 35px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD EMOTION MODEL
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
# GROK AI
# ============================================================

@st.cache_resource
def get_grok_client():

    try:

        from xai_sdk import Client

        # First try Streamlit Secrets
        key = st.secrets.get(
            "XAI_API_KEY",
            None
        )

        # If not found, try environment variable
        if not key:
            key = os.environ.get(
                "XAI_API_KEY"
            )

        if not key:
            return None

        return Client(
            api_key=key
        )

    except Exception:
        return None


# ============================================================
# ASK GROK
# ============================================================

def ask_ai(message, emotion):

    client = get_grok_client()

    # --------------------------------------------------------
    # API KEY NOT FOUND
    # --------------------------------------------------------

    if client is None:

        return (
            "I'm connected to the Emotion AI app, "
            "but the Grok API key has not been configured yet. "
            "Please add XAI_API_KEY in Streamlit Secrets."
        )


    # --------------------------------------------------------
    # GROK INSTRUCTIONS
    # --------------------------------------------------------

    system_instruction = """
You are Emotion AI, a friendly, intelligent and natural
AI assistant inside a facial emotion recognition application.

Your job is to have a normal conversation with the user while
also understanding the facial emotion detected by the application.

IMPORTANT RULES:

1. Answer the user's actual question.

2. Do not give the same generic response repeatedly.

3. If the user asks a normal question, answer the question normally.

4. If the user asks about their detected emotion, explain the
   detected facial expression clearly.

5. The detected emotion is only an AI prediction based on facial
   expression. It does NOT prove exactly how the person feels.

6. If the detected emotion is happy, you can respond positively.

7. If the detected emotion is sad, be gentle and supportive.

8. If the detected emotion is angry, respond calmly.

9. If the detected emotion is fearful, be reassuring.

10. If the detected emotion is surprised, respond naturally.

11. If the detected emotion is disgust, respond naturally and politely.

12. If the detected emotion is neutral, explain that the model
    detected a neutral facial expression.

13. Give practical general advice when appropriate.

14. Keep answers concise enough for a chatbot.

15. Be friendly and conversational.

16. Use emojis occasionally, but do not overuse them.

17. Never diagnose mental-health conditions based on facial emotion.

18. Never claim that facial emotion recognition can know someone's
    exact internal feelings.

19. If the user says "hi", "hello", "hey", etc., respond naturally
    instead of talking about their emotion immediately.

20. If the user asks for motivation, encouragement or general advice,
    provide useful and supportive advice.
"""


    # --------------------------------------------------------
    # USER MESSAGE + EMOTION CONTEXT
    # --------------------------------------------------------

    user_prompt = f"""
Current detected facial emotion:
{emotion}

Remember:
This emotion is only a machine-learning prediction of facial
expression and may not represent the person's actual feelings.

User's message:
{message}
"""


    # --------------------------------------------------------
    # CALL GROK
    # --------------------------------------------------------

    try:

        from xai_sdk.chat import (
            system,
            user
        )

        chat = client.chat.create(
            model="grok-4.6"
        )

        chat.append(
            system(
                system_instruction
            )
        )

        chat.append(
            user(
                user_prompt
            )
        )

        response = chat.sample()

        reply = response.content

        if reply:
            return reply

        return (
            "I couldn't generate a response right now. "
            "Please try again."
        )

    except Exception as e:

        return (
            f"Grok connection error: {e}"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

    <div style="
        color:#c084fc;
        font-weight:700;
        letter-spacing:2px;
    ">
        ✦ AI POWERED
    </div>

    <h1>
        Face Emotion
        <span class="gradient">
            Recognition
        </span>
    </h1>

    <div class="subtitle">
        Upload a facial image, detect the emotion,
        and chat with your AI assistant.
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT SIDE — IMAGE ANALYSIS
# ============================================================

with left:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📷 Analyze an Image"
    )

    uploaded = st.file_uploader(
        "Choose a clear face image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp"
        ],
    )

    if uploaded:

        image = Image.open(
            uploaded
        )

        st.image(
            image,
            caption="Selected image",
            use_container_width=True
        )

        if st.button(
            "🧠 Analyze Emotion",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing facial expression..."
            ):

                emotion, confidence, probabilities = (
                    predict_emotion(image)
                )

            st.session_state["emotion"] = emotion

            st.session_state["confidence"] = (
                confidence
            )

            st.session_state["probabilities"] = (
                probabilities
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT SIDE — EMOTION RESULT
# ============================================================

with right:

    emotion = st.session_state.get(
        "emotion"
    )

    if emotion:

        confidence = st.session_state.get(
            "confidence",
            0
        )

        st.markdown(
            f"""
<div class="card emotion-box">

    <div class="small">
        Detected Emotion
    </div>

    <div class="emotion-icon">
        {EMOTION_ICONS.get(emotion, "🤖")}
    </div>

    <div class="emotion-name">
        {emotion}
    </div>

    <div class="small">
        Confidence: {confidence:.1%}
    </div>

</div>
""",
            unsafe_allow_html=True
        )


        probabilities = st.session_state.get(
            "probabilities"
        )

        if probabilities is not None:

            st.markdown(
                "### 📊 Emotion Probabilities"
            )

            for name, probability in zip(
                CLASS_NAMES,
                probabilities
            ):

                st.progress(
                    float(probability),
                    text=(
                        f"{name.capitalize()} "
                        f"— {probability:.1%}"
                    )
                )

    else:

        st.markdown(
            """
<div class="card emotion-box">

    <div class="emotion-icon">
        🤖
    </div>

    <div class="emotion-name">
        Waiting for an image
    </div>

    <div class="small">
        Upload an image and click
        Analyze Emotion.
    </div>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# CHATBOT
# ============================================================

st.markdown(
    "## 💬 Emotion AI Chatbot"
)


current_emotion = st.session_state.get(
    "emotion",
    "unknown"
)


# ============================================================
# CURRENT EMOTION CARD
# ============================================================

st.markdown(
    f"""
<div class="card">

    <div class="small">
        Current detected emotion
    </div>

    <strong>
        {EMOTION_ICONS.get(
            current_emotion,
            "🤖"
        )}
        {current_emotion.capitalize()}
    </strong>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content":
                "Hi! 👋 I'm your Emotion AI assistant. "
                "Upload an image first, or just start "
                "chatting with me."
        }

    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "assistant":

        st.markdown(
            f"""
<div class="bot">

    🤖 <strong>
        Emotion AI
    </strong>

    <br>

    {message["content"]}

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div class="user">

    👤 <strong>
        You
    </strong>

    <br>

    {message["content"]}

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Type your message..."
)


if prompt:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Ask Grok
    with st.spinner(
        "Emotion AI is typing..."
    ):

        reply = ask_ai(
            prompt,
            current_emotion
        )


    # Add Grok response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )


    # Refresh app
    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

    Face Emotion Recognition
    • MobileNetV2
    + Streamlit
    + Grok AI

</div>
""",
    unsafe_allow_html=True
)
