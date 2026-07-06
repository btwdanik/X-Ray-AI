import base64
from io import BytesIO
from typing import List
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms


CHECKPOINT_PATH = r"model\densenet121_xray_512_best.pth"
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

CLASSES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]


app = FastAPI(title="X-Ray AI", version="1.0.0")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_model(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def load_model() -> nn.Module:
    model = build_model(num_classes=len(CLASSES))

    model_path = BASE_DIR / "model" / "densenet121_xray_512_best.pth"
    checkpoint = torch.load(model_path, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict)

    print("Device:", DEVICE)
    print("Model loaded:", model_path)

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)

def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def array_to_base64_rgba(array: np.ndarray) -> str:
    image = Image.fromarray(array.astype(np.uint8), mode="RGBA")
    return pil_to_base64(image)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(self.save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()

        logits = self.model(input_tensor)

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, logits

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def make_heatmap_rgba(image: Image.Image, cam: np.ndarray) -> np.ndarray:
    image_np = np.array(image.convert("RGB"))
    h, w = image_np.shape[:2]

    cam_resized = cv2.resize(cam, (w, h)).astype(np.float32)

    cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=8, sigmaY=8)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    alpha = np.clip((cam_resized - 0.15) / 0.85, 0, 1)
    alpha = np.power(alpha, 0.8)
    alpha = np.uint8(alpha * 255)

    heatmap_rgba = np.dstack([heatmap, alpha])
    return heatmap_rgba

def predict_probs(input_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    return probs

def get_top5(probs: np.ndarray) -> List[dict]:
    indices = np.argsort(probs)[::-1][:5]

    result = []

    for idx in indices:
        result.append({
            "class_idx": int(idx),
            "name": CLASSES[idx],
            "probability": round(float(probs[idx] * 100), 2),
        })

    return result

def run_pipeline(image: Image.Image) -> dict:
    input_tensor = image_to_tensor(image)

    probs = predict_probs(input_tensor)
    top5 = get_top5(probs)

    best_class_idx = top5[0]["class_idx"]

    gradcam = GradCAM(
        model=model,
        target_layer=model.features.denseblock4,
    )

    try:
        cam, _ = gradcam(input_tensor, best_class_idx)
    finally:
        gradcam.remove()

    heatmap_rgba = make_heatmap_rgba(image, cam)

    return {
        "image": pil_to_base64(image),
        "heatmap": array_to_base64_rgba(heatmap_rgba),
        "predictions": top5,
    }

@app.get("/")
def home():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только JPG и PNG",
        )

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Не удалось прочитать изображение",
        )

    try:
        return run_pipeline(image)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
