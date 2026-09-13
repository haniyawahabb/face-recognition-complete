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
# IMAGE + EMOTION SECTION
# ============================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT — UPLOAD
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

            st.session_state["confidence"] = confidence

            st.session_state["probabilities"] = probabilities

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT — EMOTION RESULT
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
        {EMOTION_ICONS.get(
            emotion,
            "🤖"
        )}
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
# CHATBOT HEADER
# ============================================================

st.markdown(
    '<h2 class="chat-title">💬 Emotion AI Chatbot</h2>',
    unsafe_allow_html=True
)


# ============================================================
# CURRENT EMOTION
# ============================================================

current_emotion = st.session_state.get(
    "emotion",
    "unknown"
)

emotion_display = current_emotion.capitalize()

st.info(
    f"{EMOTION_ICONS.get(current_emotion, '🤖')} "
    f"Current detected emotion: **{emotion_display}**"
)


# ============================================================
# CHATBOT SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Emotion AI, a friendly, intelligent and natural AI
assistant inside a Facial Emotion Recognition application.

You are a REAL conversational chatbot.

You should:
- Have natural conversations with the user.
- Remember and use the previous messages in the conversation.
- Answer the user's actual question.
- Avoid repeating generic answers.
- Be friendly, helpful and conversational.
- Give useful explanations and examples.
- Keep normal answers reasonably concise.
- Use emojis naturally but don't overuse them.

The application has a facial emotion recognition model.

The detected facial emotion is only an AI prediction based on
facial expression. It does NOT prove the person's true internal
feelings.

If the user asks about their detected emotion:
- Explain the detected expression.
- Mention that it is only an AI prediction when appropriate.

If the detected emotion is:
happy:
    Respond positively and warmly.

sad:
    Be gentle and supportive.

angry:
    Respond calmly.

fear:
    Be reassuring.

surprise:
    Respond naturally.

disgust:
    Respond naturally and politely.

neutral:
    Explain that the model detected a neutral facial expression.

IMPORTANT:
Never diagnose mental-health conditions based on facial emotion.
Never claim that facial recognition can know exactly how someone
feels internally.

If the user asks a completely normal question, answer that
question normally instead of unnecessarily talking about emotions.
"""


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 I'm **Emotion AI**. "
                "Upload a face image to detect an emotion, "
                "or simply start chatting with me!"
            )
        }
    ]


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=(
            "🤖"
            if message["role"] == "assistant"
            else "👤"
        )
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Message Emotion AI..."
)


if prompt:

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            prompt
        )


    # --------------------------------------------------------
    # GET GROQ CLIENT
    # --------------------------------------------------------

    client = get_groq_client()


    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        if client is None:

            reply = (
                "⚠️ Groq API key is not configured yet. "
                "Please add `GROQ_API_KEY` in Streamlit Secrets."
            )

            st.error(reply)

        else:

            # ------------------------------------------------
            # BUILD CONVERSATION
            # ------------------------------------------------

            api_messages = [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                        +
                        f"""

CURRENT DETECTED FACIAL EMOTION:
{current_emotion}

Remember:
This detected emotion is only an AI prediction of facial
expression and does not guarantee the person's actual feelings.
"""
                    )
                }
            ]


            # Add previous conversation
            for message in st.session_state.messages:

                api_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                )


            # ------------------------------------------------
            # STREAM GROQ RESPONSE
            # ------------------------------------------------

            try:

                response_container = st.empty()

                full_response = ""

                for piece in generate_groq_response(
                    client,
                    api_messages
                ):

                    full_response += piece

                    response_container.markdown(
                        full_response + "▌"
                    )

                response_container.markdown(
                    full_response
                )


                # ------------------------------------------------
                # SAVE ASSISTANT RESPONSE
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response
                    }
                )


            except Exception as e:

                error_message = (
                    f"Sorry, I couldn't connect to Groq right now.\n\n"
                    f"Error: `{e}`"
                )

                st.error(
                    error_message
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

    Face Emotion Recognition
    • MobileNetV2
    • Streamlit
    • Groq AI

</div>
""",
    unsafe_allow_html=True
)
