from flask import Flask, render_template, request, jsonify, Response
import torch
import cv2
import os
import tempfile
import json
from PIL import Image
import numpy as np
from torchvision import transforms
import base64
from io import BytesIO
import threading
import time

# Import your existing modules
from model import ViolenceDetector
from utils import extract_frames

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variables for webcam detection
webcam_active = False
current_prediction = {"label": "Waiting...", "confidence": 0.0}


# Model setup
# Remove the decorator and define the function normally
def load_model():
    global model, device, transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetector().to(device)
    model_path = "checkpoints/violence_detector_best/best_model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])


def predict_clip(frames, max_frames=20):
    global model, device, transform

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

    label = {0: "Non-Violent", 1: "Violent"}[pred]
    confidence = probs[0][pred].item()

    return label, confidence


def predict_video(video_path):
    frames = extract_frames(video_path, max_frames=20)
    return predict_clip(frames)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        # Predict
        label, confidence = predict_video(temp_path)

        # Clean up
        os.unlink(temp_path)

        return jsonify({
            'label': label,
            'confidence': confidence,
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active
    data = request.get_json()
    webcam_id = data.get('webcam_id', 0)

    webcam_active = True

    # Start webcam detection in a separate thread
    thread = threading.Thread(target=webcam_detection_loop, args=(webcam_id,))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True})


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({'success': True})


@app.route('/webcam_status')
def webcam_status():
    global current_prediction
    return jsonify(current_prediction)


def webcam_detection_loop(webcam_id):
    global webcam_active, current_prediction

    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        current_prediction = {"label": "Camera Error", "confidence": 0.0}
        webcam_active = False
        return

    buffer = []
    max_frames = 20

    while webcam_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            current_prediction = {"label": "Camera Error", "confidence": 0.0}
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (224, 224))
        pil_img = Image.fromarray(resized)

        buffer.append(pil_img)

        if len(buffer) > max_frames:
            buffer.pop(0)

        if len(buffer) == max_frames:
            try:
                label, confidence = predict_clip(buffer)
                current_prediction = {"label": label, "confidence": confidence}
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                current_prediction = {"label": "Error", "confidence": 0.0}

        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

    cap.release()
    webcam_active = False


def generate_webcam_feed(webcam_id):
    global webcam_active
    cap = cv2.VideoCapture(webcam_id)

    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {webcam_id}")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() +
               b'\r\n')
        return

    while webcam_active:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from webcam {webcam_id}")
            break

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed/<int:webcam_id>')
def video_feed(webcam_id):
    print(f"Starting video feed for webcam ID {webcam_id}")
    return Response(generate_webcam_feed(webcam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    load_model()
    app.run(debug=True, threaded=True)