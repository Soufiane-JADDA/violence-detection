"""
FastAPI App for Violence Detection

POST /predict → Upload a video → Get violent/non-violent classification
"""
# pip install python-multipart fastapi uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import tempfile
import numpy as np
from torchvision import transforms
from PIL import Image
import uvicorn

# Local imports
from model import ViolenceDetector
from utils import extract_frames

# === Initialize App ===
app = FastAPI(
    title="🚨 Violence Detection API",
    description="Upload a video and get violence detection results.",
    version="1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Load Model ===
def load_model():
    model = ViolenceDetector().to(device)
    model_path = "checkpoints/violence_detector_best/best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


model = load_model()

# === Transforms (Same as during training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# === Prediction Endpoint ===
@app.post("/predict", summary="Detect violence in a video")
async def predict(file: UploadFile = File(...)):
    """
    Upload a video file (.mp4, .avi) and get a violence detection result.

    Returns:
    - prediction: "Violent" or "Non-Violent"
    - confidence: Probability of predicted class
    """

    # Save uploaded file temporarily
    suffix = os.path.splitext(file.filename)[1]
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_video.write(await file.read())
    temp_video.close()

    try:
        # Extract and transform frames
        frames = extract_frames(temp_video.name, max_frames=20)

        # Pad if necessary
        while len(frames) < 20:
            frames.append(frames[-1])

        processed_frames = [transform(Image.fromarray(f)) for f in frames]
        video_tensor = torch.stack(processed_frames)  # (T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)

        # Run model
        with torch.no_grad():
            output = model(video_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label_map = {0: "Non-Violent", 1: "Violent"}
        prediction = label_map[pred]
        confidence = probs[0][pred].item()

        return JSONResponse(
            content={
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "success": True,
            }
        )

    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "success": False}, status_code=500
        )

    finally:
        os.unlink(temp_video.name)


# === Run with Uvicorn ===
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
