// ========================================
// EMOTION AI - FRONTEND SCRIPT
// ========================================


// ========================================
// IMAGE ELEMENTS
// ========================================

const imageInput = document.getElementById("imageInput");
const chooseBtn = document.getElementById("chooseBtn");

const previewContainer =
    document.getElementById("previewContainer");

const previewImage =
    document.getElementById("previewImage");

const removeBtn =
    document.getElementById("removeBtn");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const resultCard =
    document.getElementById("resultCard");

const emotionText =
    document.getElementById("emotionText");

const emotionIcon =
    document.getElementById("emotionIcon");

const confidenceText =
    document.getElementById("confidenceText");

const progressBar =
    document.getElementById("progressBar");

const uploadArea =
    document.getElementById("uploadArea");


// ========================================
// CHATBOT ELEMENTS
// ========================================

const chatbotToggle =
    document.getElementById("chatbotToggle");

const chatbotWindow =
    document.getElementById("chatbotWindow");

const closeChat =
    document.getElementById("closeChat");

const chatMessages =
    document.getElementById("chatMessages");

const chatInput =
    document.getElementById("chatInput");

const sendMessage =
    document.getElementById("sendMessage");

const typingIndicator =
    document.getElementById("typingIndicator");

const quickActions =
    document.getElementById("quickActions");

const emotionContext =
    document.getElementById("emotionContext");

const emotionContextText =
    document.getElementById("emotionContextText");


// ========================================
// FASTAPI BACKEND URL
// ========================================

const API_BASE_URL =
    "https://face-recognition-complete.fastapicloud.dev";

const PREDICT_API_URL =
    `${API_BASE_URL}/predict`;

const CHAT_API_URL =
    `${API_BASE_URL}/chat`;


// ========================================
// CURRENT DETECTED EMOTION
// ========================================

let currentEmotion = null;


// ========================================
// EMOTION ICONS
// ========================================

const emotionIcons = {

    angry: "😠",
    disgust: "🤢",
    fear: "😨",
    happy: "😊",
    neutral: "😐",
    sad: "😢",
    surprise: "😲"

};


// ========================================
// CHOOSE IMAGE
// ========================================

chooseBtn.addEventListener("click", () => {

    imageInput.click();

});


// ========================================
// IMAGE SELECTED
// ========================================

imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }


    // Check image type

    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image.");

        imageInput.value = "";

        return;
    }


    // Preview image

    const reader = new FileReader();


    reader.onload = function (event) {

        previewImage.src =
            event.target.result;


        // Hide upload area

        uploadArea.style.display =
            "none";


        // Show preview

        previewContainer.style.display =
            "block";


        // Hide old result

        resultCard.style.display =
            "none";


        // Reset progress

        progressBar.style.width =
            "0%";


        // Reset current emotion

        currentEmotion = null;

    };


    reader.readAsDataURL(file);

});


// ========================================
// REMOVE IMAGE
// ========================================

removeBtn.addEventListener("click", () => {

    imageInput.value = "";

    previewImage.src = "";


    previewContainer.style.display =
        "none";


    uploadArea.style.display =
        "block";


    resultCard.style.display =
        "none";


    progressBar.style.width =
        "0%";


    emotionText.textContent =
        "";

    confidenceText.textContent =
        "";

    emotionIcon.textContent =
        "";


    // Reset detected emotion

    currentEmotion = null;

});


// ========================================
// ANALYZE EMOTION
// ========================================

analyzeBtn.addEventListener("click", async () => {

    const file = imageInput.files[0];


    // No image selected

    if (!file) {

        alert(
            "Please choose an image first."
        );

        return;
    }


    // ========================================
    // LOADING STATE
    // ========================================

    analyzeBtn.disabled = true;

    analyzeBtn.textContent =
        "Analyzing...";


    try {

        // ========================================
        // CREATE FORM DATA
        // ========================================

        const formData =
            new FormData();


        // FastAPI expects "file"

        formData.append(
            "file",
            file
        );


        // ========================================
        // SEND IMAGE TO FASTAPI
        // ========================================

        const response =
            await fetch(
                PREDICT_API_URL,
                {
                    method: "POST",
                    body: formData
                }
            );


        // ========================================
        // CHECK RESPONSE
        // ========================================

        if (!response.ok) {

            let errorMessage =
                "Prediction failed.";

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {

                    errorMessage =
                        errorData.detail;

                }

            } catch (error) {

                console.log(
                    "Could not read error response."
                );

            }


            throw new Error(
                `${errorMessage} (${response.status})`
            );
        }


        // ========================================
        // GET JSON RESPONSE
        // ========================================

        const data =
            await response.json();


        console.log(
            "FastAPI Response:",
            data
        );


        // ========================================
        // GET PREDICTED EMOTION
        // ========================================

        const emotion =
            data.predicted_emotion;


        if (!emotion) {

            throw new Error(
                "No emotion was returned by the API."
            );
        }


        // ========================================
        // SAVE CURRENT EMOTION
        // ========================================

        currentEmotion =
            emotion.toLowerCase();


        // ========================================
        // GET CONFIDENCE
        // ========================================

        let confidence =
            Number(data.confidence);


        if (Number.isNaN(confidence)) {

            confidence = 0;

        }


        // Backend confidence is 0-1

        confidence =
            confidence * 100;


        // Keep between 0 and 100

        confidence =
            Math.max(
                0,
                Math.min(
                    100,
                    confidence
                )
            );


        // ========================================
        // EMOTION NAME
        // ========================================

        const emotionName =
            emotion.charAt(0).toUpperCase() +
            emotion.slice(1).toLowerCase();


        // ========================================
        // EMOTION ICON
        // ========================================

        const icon =
            emotionIcons[
                emotion.toLowerCase()
            ] || "🙂";


        // ========================================
        // UPDATE RESULT UI
        // ========================================

        emotionText.textContent =
            emotionName;


        emotionIcon.textContent =
            icon;


        confidenceText.textContent =
            confidence.toFixed(2) + "%";


        // Show result card

        resultCard.style.display =
            "block";


        // ========================================
        // ANIMATE PROGRESS BAR
        // ========================================

        progressBar.style.width =
            "0%";


        setTimeout(() => {

            progressBar.style.width =
                confidence + "%";

        }, 100);


        // ========================================
        // UPDATE CHATBOT EMOTION CONTEXT
        // ========================================

        updateChatbotEmotion(
            emotionName,
            icon
        );


        // ========================================
        // OPEN CHATBOT AUTOMATICALLY
        // ========================================

        setTimeout(() => {

            if (chatbotWindow) {

                chatbotWindow.classList.add(
                    "active"
                );

            }

        }, 500);


        // ========================================
        // LOG PROBABILITIES
        // ========================================

        console.log(
            "Emotion probabilities:",
            data.probabilities
        );


    } catch (error) {


        // ========================================
        // ERROR HANDLING
        // ========================================

        console.error(
            "Prediction Error:",
            error
        );


        alert(
            "Unable to analyze the image.\n\n" +
            error.message
        );


        resultCard.style.display =
            "none";


    } finally {


        // ========================================
        // RESET BUTTON
        // ========================================

        analyzeBtn.disabled =
            false;


        analyzeBtn.textContent =
            "Analyze Emotion";

    }

});


// ========================================
// CHATBOT OPEN
// ========================================

if (chatbotToggle) {

    chatbotToggle.addEventListener(
        "click",
        () => {

            chatbotWindow.classList.add(
                "active"
            );

            setTimeout(() => {

                if (chatInput) {
                    chatInput.focus();
                }

            }, 200);

        }
    );

}


// ========================================
// CHATBOT CLOSE
// ========================================

if (closeChat) {

    closeChat.addEventListener(
        "click",
        () => {

            chatbotWindow.classList.remove(
                "active"
            );

        }
    );

}


// ========================================
// UPDATE EMOTION CONTEXT
// ========================================

function updateChatbotEmotion(
    emotionName,
    icon
) {

    if (!emotionContext ||
        !emotionContextText) {

        return;
    }


    emotionContextText.textContent =
        `I detected ${emotionName} ${icon} from your image. I'm here if you'd like to talk about how you're feeling.`;


    emotionContext.style.display =
        "flex";

}


// ========================================
// ADD USER MESSAGE
// ========================================

function addUserMessage(message) {

    const messageElement =
        document.createElement("div");


    messageElement.className =
        "message user-message";


    messageElement.innerHTML = `

        <div class="message-content">

            <p>${escapeHTML(message)}</p>

        </div>

    `;


    chatMessages.appendChild(
        messageElement
    );


    scrollChatToBottom();

}


// ========================================
// ADD BOT MESSAGE
// ========================================

function addBotMessage(message) {

    const messageElement =
        document.createElement("div");


    messageElement.className =
        "message bot-message";


    messageElement.innerHTML = `

        <div class="message-avatar">
            🤖
        </div>

        <div class="message-content">

            <p>${formatBotMessage(message)}</p>

        </div>

    `;


    chatMessages.appendChild(
        messageElement
    );


    scrollChatToBottom();

}


// ========================================
// FORMAT BOT MESSAGE
// ========================================

function formatBotMessage(message) {

    return escapeHTML(message)
        .replace(/\n/g, "<br>");

}


// ========================================
// ESCAPE HTML
// ========================================

function escapeHTML(text) {

    const div =
        document.createElement("div");

    div.textContent =
        text;

    return div.innerHTML;

}


// ========================================
// SCROLL CHAT
// ========================================

function scrollChatToBottom() {

    if (!chatMessages) {
        return;
    }

    chatMessages.scrollTop =
        chatMessages.scrollHeight;

}


// ========================================
// SHOW TYPING
// ========================================

function showTyping() {

    if (!typingIndicator) {
        return;
    }

    typingIndicator.style.display =
        "flex";

    scrollChatToBottom();

}


// ========================================
// HIDE TYPING
// ========================================

function hideTyping() {

    if (!typingIndicator) {
        return;
    }

    typingIndicator.style.display =
        "none";

}


// ========================================
// SEND CHAT MESSAGE
// ========================================

async function handleSendMessage(message = null) {

    const text =
        message !== null
            ? message.trim()
            : chatInput.value.trim();


    // Empty message

    if (!text) {
        return;
    }


    // Show user message

    addUserMessage(text);


    // Clear input

    chatInput.value = "";


    // Disable send

    if (sendMessage) {
        sendMessage.disabled = true;
    }


    // Show typing

    showTyping();


    try {

        // ========================================
        // SEND MESSAGE TO BACKEND
        // ========================================

        const response =
            await fetch(
                CHAT_API_URL,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        message: text,

                        emotion:
                            currentEmotion || "unknown"

                    })

                }
            );


        // ========================================
        // CHECK RESPONSE
        // ========================================

        if (!response.ok) {

            let errorMessage =
                "Chat request failed.";

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {

                    errorMessage =
                        errorData.detail;

                }

            } catch (error) {

                console.log(
                    "Could not read chat error."
                );

            }


            throw new Error(
                `${errorMessage} (${response.status})`
            );

        }


        // ========================================
        // GET CHAT RESPONSE
        // ========================================

        const data =
            await response.json();


        console.log(
            "Chatbot Response:",
            data
        );


        // Support common response names

        const reply =
            data.response ||
            data.reply ||
            data.message;


        if (!reply) {

            throw new Error(
                "No chatbot response was returned."
            );

        }


        // Hide typing

        hideTyping();


        // Show bot response

        addBotMessage(reply);


    } catch (error) {

        console.error(
            "Chat Error:",
            error
        );


        hideTyping();


        addBotMessage(
            "Sorry, I'm having trouble connecting right now. Please try again in a moment."
        );


    } finally {

        // Enable send

        if (sendMessage) {
            sendMessage.disabled = false;
        }


        if (chatInput) {
            chatInput.focus();
        }

    }

}


// ========================================
// SEND BUTTON
// ========================================

if (sendMessage) {

    sendMessage.addEventListener(
        "click",
        () => {

            handleSendMessage();

        }
    );

}


// ========================================
// ENTER KEY
// ========================================

if (chatInput) {

    chatInput.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                handleSendMessage();

            }

        }
    );

}


// ========================================
// QUICK ACTION BUTTONS
// ========================================

if (quickActions) {

    const quickButtons =
        quickActions.querySelectorAll(
            ".quick-btn"
        );


    quickButtons.forEach(
        (button) => {

            button.addEventListener(
                "click",
                () => {

                    const message =
                        button.dataset.message;


                    if (message) {

                        handleSendMessage(
                            message
                        );

                    }

                }
            );

        }
    );

}
