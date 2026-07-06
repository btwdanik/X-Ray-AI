const API_URL = "/predict";

const fileInput = document.getElementById("fileInput");
const xrayImage = document.getElementById("xrayImage");
const heatmapLayer = document.getElementById("heatmapLayer");
const emptyState = document.getElementById("emptyState");
const camSlider = document.getElementById("camSlider");
const resultsPanel = document.getElementById("resultsPanel");
const imageWrapper = document.getElementById("imageWrapper");

const defaultPredictions = [
    { name: "No finding", probability: 0 },
    { name: "Cardiomegaly", probability: 0 },
    { name: "Pleural effusion", probability: 0 },
    { name: "Lung Opacity", probability: 0 },
    { name: "Pulmonary fibrosis", probability: 0 },
];

const colors = [
    ["#7f1d1d", "#ef4444"],
    ["#991b1b", "#f43f5e"],
    ["#b91c1c", "#ef4444"],
    ["#dc2626", "#fb7185"],
    ["#ef4444", "#ff003c"],
];

function formatPercent(value) {
    const probability = Number(value) || 0;

    if (probability >= 10) return `${Math.round(probability)}%`;
    if (probability >= 1) return `${probability.toFixed(1)}%`;
    if (probability > 0) return "<1%";

    return "0%";
}

function getVisualWidth(value) {
    const probability = Number(value) || 0;
    return probability > 0 ? Math.max(probability, 4) : 0;
}

function renderResults(predictions = defaultPredictions) {
    resultsPanel.innerHTML = predictions.slice(0, 5).map((item, index) => {
        const probability = Number(item.probability) || 0;
        const [startColor, endColor] = colors[index] || colors[0];
        const topClass = index === 0 ? "top" : "";

        return `
            <div class="result-card ${topClass}">
                <div class="result-header">
                    <span>${item.name}</span>
                    <b>${formatPercent(probability)}</b>
                </div>

                <div class="progress">
                    <div
                        class="progress-fill"
                        style="
                            width: ${getVisualWidth(probability)}%;
                            background: linear-gradient(90deg, ${startColor}, ${endColor});
                        "
                    ></div>
                </div>
            </div>
        `;
    }).join("");
}

function updateHeatmapOpacity() {
    heatmapLayer.style.opacity = String(Number(camSlider.value) / 100);
}

function setHeatmap(url = "") {
    heatmapLayer.style.backgroundImage = url ? `url("${url}")` : "";
    heatmapLayer.style.display = url ? "block" : "none";
    heatmapLayer.style.opacity = url ? String(Number(camSlider.value) / 100) : "0";
}

function showPreview(file) {
    xrayImage.src = URL.createObjectURL(file);
    xrayImage.style.display = "block";
    emptyState.style.display = "none";
    setHeatmap();
}

function showResult(result) {
    xrayImage.src = result.image;
    xrayImage.style.display = "block";
    emptyState.style.display = "none";

    setHeatmap(result.heatmap);
    renderResults(result.predictions);
}

async function sendImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        return response.json();
    }

    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || "Backend error when processing the snapshot");
}

async function handleFileUpload() {
    const file = fileInput.files[0];
    if (!file) return;

    showPreview(file);
    renderResults();
    imageWrapper.classList.add("is-loading");

    try {
        const result = await sendImage(file);
        showResult(result);
    } catch (error) {
        alert(error.message);
    } finally {
        imageWrapper.classList.remove("is-loading");
    }
}

fileInput.addEventListener("change", handleFileUpload);
camSlider.addEventListener("input", updateHeatmapOpacity);

renderResults();
updateHeatmapOpacity();
