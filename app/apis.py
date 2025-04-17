import os
import requests

# Download classification model if not already present
def download_model(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading model from {url}...")
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

# Replace this with the actual file name you uploaded in GitHub Release
download_model(
    "https://github.com/shawky-gamal-22/Classification_Segmentation_APIs/releases/download/v1.0/unet_80.pth", "model_weights/unet_80.pth"
)

# ----------------------------------------------------------------------------------------
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import pickle
import io
from typing import Dict
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import zipfile
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# -------------------------------- Classification Model ------------------------------------

# Load idx_to_name
idx_to_name = pickle.load(open('idx_to_name.pkl', 'rb'))

# Classification Model
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, num_classes
        )
    
    def forward(self, x):
        return self.base_model(x)

num_classes = 4
model = EfficientNetModel(num_classes).to(device)
model.load_state_dict(torch.load('model_weights/efficientnet_model.pth', map_location=device))
model.eval()

transforms_classify = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classification_model(image: Image.Image) -> Dict:
    image = transforms_classify(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        output = torch.softmax(output, axis=1)
        idx = torch.argmax(output, axis=1).item()
        label = idx_to_name[idx]
        confidence = torch.max(output, axis=1).values.item()
    return {"label": label, "confidence": confidence}

# -------------------------------- Segmentation Model --------------------------------------

IMG_SIZE = 640
transforms_segmenation = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

# Segmentation model
NUM_CLASSES = 5  # Optic Disc, Microaneurysms, Hemorrhages, Soft Exudates, Hard Exudates
unet_model = smp.Unet(
    encoder_name="resnet34",  
    encoder_weights="imagenet",  
    in_channels=3,  
    classes=NUM_CLASSES
)
unet_model.load_state_dict(torch.load("model_weights/unet_80.pth", map_location=device))
unet_model.to(device)
unet_model.eval()

def preprocess(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = transforms_segmenation(image=img)
    tensor = transformed["image"].unsqueeze(0).float().to(device)
    return tensor

def segmentation_model(image: Image.Image):
    image = preprocess(image)
    with torch.no_grad():
        output = unet_model(image)
        output = torch.sigmoid(output)
        masks = (output > 0.3).float()
        masks = masks.squeeze().cpu().numpy()
        masks = masks.transpose(1, 2, 0)
        masks = (masks * 255).astype("uint8")
    return masks

# ------------------------------- Routes ----------------------------------------------------

@app.post("/predict_classification/")
async def classify(file: UploadFile = File(...)):
    if file.content_type.startswith("image/"):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classification_model(image)
        return result
    return JSONResponse(content={"error": "Invalid file type"}, status_code=400)

@app.post("/predict_segmentation/")
async def segment(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    masks = segmentation_model(image)

    disease_names = [
        "optic_disc",
        "microaneurysms",
        "hemorrhages",
        "soft_exudates",
        "hard_exudates"
    ]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, name in enumerate(disease_names):
            channel = masks[:, :, i]
            channel = np.clip(channel, 0, 255).astype("uint8")
            pil_image = Image.fromarray(channel).convert("L")
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            zip_file.writestr(f"{name}.png", img_byte_arr.read())

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=disease_masks.zip"})
