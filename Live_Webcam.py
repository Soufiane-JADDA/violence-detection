import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import transforms

# === Local imports ===
from model import ViolenceDetector
from utils import extract_frames

# === Webcam Live Detection ===
st.title("📷 Live Violence Detection")
st.markdown("Select your webcam and begin live detection.")

st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.header("About")
st.sidebar.write("""
This app uses a deep learning model (CNN + LSTM + Attention) to classify videos as **Violent** or **Non-Violent**.
Built by AI Research Team.
""")

if st.button("Start Webcam Detection"):
    model, device = load_model()
    stframe = st.empty()
    camera_index = st.sidebar.selectbox("Select Webcam", options=list(range(5)), index=0)
    cap = cv2.VideoCapture(camera_index)

    buffer = []
    frame_count = 0
    max_frames = 20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (224, 224))
        tensor = transform(Image.fromarray(resized)).unsqueeze(0)
        buffer.append(tensor)
        frame_count += 1

        if len(buffer) > max_frames:
            buffer.pop(0)

        stframe.image(img, channels="RGB", caption="Live Feed", use_container_width=True)

        if len(buffer) == max_frames:
            label, confidence = predict_clip(model, device, [t.squeeze().permute(1, 2, 0).mul(0.5).add(0.5).clamp(0, 1).mul(255).byte().cpu().numpy() for t in buffer])
            stframe.markdown(f"### Prediction: **{label}** - Confidence: {confidence*100:.2f}%")

        if frame_count > 300:
            break

    cap.release()
