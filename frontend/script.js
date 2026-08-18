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
// FASTAPI BACKEND URL
// ========================================

const API_URL =
    "https://face-recognition-complete.fastapicloud.dev/predict";


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


        // IMPORTANT:
        // FastAPI expects field name "file"

        formData.append(
            "file",
            file
        );


        // ========================================
        // SEND IMAGE TO FASTAPI
        // ========================================

        const response =
            await fetch(
                API_URL,
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
        // GET CONFIDENCE
        // ========================================

        let confidence =
            Number(data.confidence);


        if (Number.isNaN(confidence)) {

            confidence = 0;

        }


        // Backend confidence is 0-1
        // Convert to percentage

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
        // UPDATE UI
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
        // OPTIONAL:
        // SHOW ALL PROBABILITIES
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
