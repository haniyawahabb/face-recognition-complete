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


/* CHOOSE IMAGE */

chooseBtn.addEventListener("click", () => {
    imageInput.click();
});


/* IMAGE SELECTED */

imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) return;

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src = event.target.result;

        document.getElementById("uploadArea").style.display =
            "none";

        previewContainer.style.display = "block";

        resultCard.style.display = "none";
    };

    reader.readAsDataURL(file);
});


/* REMOVE IMAGE */

removeBtn.addEventListener("click", () => {

    imageInput.value = "";

    previewImage.src = "";

    previewContainer.style.display = "none";

    document.getElementById("uploadArea").style.display =
        "block";

    resultCard.style.display = "none";
});


/* ANALYZE */

analyzeBtn.addEventListener("click", async () => {

    if (!imageInput.files[0]) {
        alert("Please choose an image first.");
        return;
    }

    analyzeBtn.textContent = "Analyzing...";
    analyzeBtn.disabled = true;


    /*
       Temporary demo result.

       Baad mein isi jagah tumhare FastAPI
       /predict endpoint ko connect karenge.
    */

    setTimeout(() => {

        const emotions = [
            {
                name: "Happy",
                icon: "😊",
                confidence: 94
            },
            {
                name: "Sad",
                icon: "😢",
                confidence: 87
            },
            {
                name: "Angry",
                icon: "😠",
                confidence: 91
            },
            {
                name: "Neutral",
                icon: "😐",
                confidence: 89
            }
        ];

        const result =
            emotions[Math.floor(Math.random() * emotions.length)];


        emotionText.textContent = result.name;

        emotionIcon.textContent = result.icon;

        confidenceText.textContent =
            result.confidence + "%";

        resultCard.style.display = "block";


        setTimeout(() => {
            progressBar.style.width =
                result.confidence + "%";
        }, 100);


        analyzeBtn.textContent = "Analyze Emotion";

        analyzeBtn.disabled = false;

    }, 1000);

});