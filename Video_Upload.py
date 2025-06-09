import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import transforms
import sys
import os

# Add the parent directory (root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Local imports ===
from model import ViolenceDetector
from utils import extract_frames

# === Page Config ===
st.set_page_config(page_title="Violence Detection", layout="centered")

# === CSS Styling ===
st.markdown("""
<style>
body {
    background-color: #f9f9f9;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #d62728;
    text-align: center;
}
.upload-box {
    border: 2px dashed #ccc;
    padding: 20px;
    border-radius: 10px;
    background-color: #fff;
}
button {
    background-color: #d62728;
    color: white;
    border-radius: 8px;
</style>
""", unsafe_allow_html=True)

# === Load model and set device ===
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetector().to(device)
    model_path = "checkpoints/violence_detector_best/best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_clip(model, device, frames, max_frames=20):
    if not frames:
        return "No Frames", 0.0

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

    label_map = {0: "Non-Violent", 1: "Violent"}
    return label_map[pred], probs[0][pred].item()

def predict_video(model, device, video_path):
    frames = extract_frames(video_path, max_frames=20)
    return predict_clip(model, device, frames)

# === UI ===
st.title("Violence Detection App")
st.markdown("Upload a video file or use webcam to detect violence.")

# Sidebar
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.header("About")
st.sidebar.write("""
This app uses a deep learning model (CNN + LSTM + Attention) to classify videos as **Violent** or **Non-Violent**.
Built by AI Research Team.
""")

uploaded_file = st.file_uploader("Upload a video (.mp4, .avi)", type=["mp4", "avi"])

if uploaded_file is not None:
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(temp_path)
    with col2:
        st.markdown("### Video Info")
        st.write(f"Name: `{uploaded_file.name}`")
        st.write(f"Type: `{uploaded_file.type}`")

    if st.button("Analyze Video"):
        model, device = load_model()
        with st.spinner("Analyzing..."):
            label, confidence = predict_video(model, device, temp_path)

        result_color = "🟢" if label == "Non-Violent" else "🔴"
        st.markdown(f"## Result: {result_color} **{label}**")
        st.progress(int(confidence * 100))
        st.metric(label="Confidence", value=f"{int(confidence * 100)}%")

        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        if ret:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Sample Frame", use_container_width=True)
        cap.release()
        os.remove(temp_path)
else:
    st.info("Please upload a video to begin.")

