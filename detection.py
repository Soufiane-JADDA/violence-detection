import cv2
import torch
from torchvision import transforms
from model import ViolenceDetector
from collections import deque
from playsound import playsound
import threading  # So it doesn't block the frame display

def play_alert_sound():
    playsound("police.wav")

# Load classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Device:", device)

# === Load trained model ===
model = ViolenceDetector().to(device)
checkpoint = torch.load("checkpoints/violence_detector_best.pt", map_location=device)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# === Frame preprocessing ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def preprocess_frames(frames, max_len=20):
    # Keep only the last `max_len` frames
    if len(frames) > max_len:
        frames = frames[-max_len:]
    while len(frames) < max_len:
        frames.insert(0, frames[0])  # Duplicate first frame to pad
    # Convert all to tensor
    tensors = [transform(f).unsqueeze(0) for f in frames]
    return torch.cat(tensors, dim=0).unsqueeze(0).to(device)  # (1, T, C, H, W)

# === Webcam ===
cap = cv2.VideoCapture(0)
frame_buffer = []
font = cv2.FONT_HERSHEY_SIMPLEX
cooldown = 0

# Maintain last N predictions
history_queue = deque(maxlen=15)  # You can adjust the window size
alert_trigger_threshold = 12      # Trigger alert if ≥ 3/5 predictions are violent


print("🟢 Real-time Violence Detection Started (press 'q' to quit).")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buffer.append(rgb_frame)

    if len(frame_buffer) > 20:
        frame_buffer = frame_buffer[-20:]

    prob = 0.0
    label_text = "Initializing..."
    color = (255, 255, 255)

    if len(frame_buffer) >= 20:
        clip = preprocess_frames(frame_buffer)
        with torch.no_grad():
            output = model(clip)
            prob = torch.softmax(output, dim=1)[0][1].item()
            pred = torch.argmax(output, dim=1).item()

        # Append 1 if violent and confident, else 0
        history_queue.append(1 if prob > 0.80 and pred == 1 else 0)

        if sum(history_queue) >= alert_trigger_threshold:
            label_text = "🚨 ALERT: Violence Detected"
            color = (0, 0, 255)
            if cooldown == 0:
                threading.Thread(target=play_alert_sound).start()
                cooldown = 30
            if cooldown > 0:
                cooldown -= 1
        else:
            label_text = f"🟢 Normal (Violence: {prob:.2f})"
            color = (0, 255, 0)

        print(f"Prob: {prob:.2f}, Pred: {pred}, Queue: {list(history_queue)}")


    cv2.putText(frame, label_text, (10, 40), font, 0.7, color, 2)
    cv2.imshow("Violence Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()