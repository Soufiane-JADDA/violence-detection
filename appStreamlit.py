import streamlit as st
import sys
import os

from model import ViolenceDetector
from utils import extract_frames  # Assuming these exist in root

# === App Configuration ===
st.set_page_config(page_title="Violence Detection", layout="centered")

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📼 Video Upload", "🎥 Live Detection"])

# === Shared Model & Transform Setup ===
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetector().to(device)
    model_path = "checkpoints/violence_detector_best/best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_clip(model, device, frames, max_frames=20):
    if len(frames) < max_frames:
        frames += [frames[0]] * (max_frames - len(frames))
    elif len(frames) > max_frames:
        frames = frames[:max_frames]
    tensors = [transform(f).unsqueeze(0) for f in frames]
    video_tensor = torch.cat(tensors, dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(video_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return {0: "Non-Violent", 1: "Violent"}[pred], probs[0][pred].item()

def predict_video(model, device, video_path):
    frames = extract_frames(video_path, max_frames=20)
    return predict_clip(model, device, frames)

# === Pages ===
if page == "🏠 Home":
    st.title("👁️ Welcome to the Violence Detection App")
    st.image("assets/logo.png", width=400)
    st.header("📖 About")
    st.write("This app uses a deep learning model (CNN + LSTM + Attention) to classify videos as Violent or Non-Violent.")
    st.markdown("### 📌 Use the sidebar to:")
    st.markdown("- 📼 Upload and analyze a video\n- 🎥 Use webcam for real-time detection")

elif page == "📼 Video Upload":
    st.title("📼 Upload Video for Violence Detection")
    uploaded_file = st.file_uploader("Upload a video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_path)
        if st.button("Analyze Video"):
            model, device = load_model()
            with st.spinner("Analyzing..."):
                label, confidence = predict_video(model, device, temp_path)
            st.success(f"**Prediction:** {label} ({confidence*100:.2f}%)")
        os.remove(temp_path)

elif page == "🎥 Live Detection":
    st.title("🎥 Real-Time Violence Detection")
    webcam_id = st.number_input("Choose Webcam ID", min_value=0, max_value=10, value=0)
    if st.button("Start Detection"):
        model, device = load_model()
        image_placeholder = st.empty()     # For webcam frame
        prediction_placeholder = st.empty()  # For predictions

        cap = cv2.VideoCapture(webcam_id)
        buffer = []
        max_frames = 20

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame from webcam.")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img, (224, 224))
            tensor = transform(Image.fromarray(resized)).unsqueeze(0)
            buffer.append(tensor)

            if len(buffer) > max_frames:
                buffer.pop(0)

            image_placeholder.image(img, caption="Live Feed", channels="RGB", use_container_width=True)

            if len(buffer) == max_frames:
                frames = [t.squeeze().permute(1, 2, 0).mul(0.5).add(0.5).clamp(0, 1).mul(255).byte().cpu().numpy() for t in buffer]
                label, conf = predict_clip(model, device, frames)
                prediction_placeholder.markdown(f"### 🔍 Prediction: **{label}** — {conf*100:.2f}%")

        cap.release()
