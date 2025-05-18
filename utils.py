import cv2
import numpy as np
import re

def extract_frames(video_path, max_frames=20, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    count = 0

    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue  # skip unreadable frames
        if count % step == 0:
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            frames.append(frame)
        count += 1

    cap.release()
    return frames if frames else [np.zeros((224, 224, 3), dtype=np.uint8)] * max_frames

def extract_epoch_num(filename):
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else -1


def save_loss_plot(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    # plt.show()

def save_losses(train_losses, val_losses, loss_log_path="loss_history.json"):
    with open(loss_log_path, "w") as file:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, file)


# Load existing loss data if available
def load_losses(loss_log_path="loss_history.json"):
    if os.path.exists(loss_log_path):
        with open(loss_log_path, "r") as f:
            history = json.load(f)
            train_losses = history.get("train_losses", [])
            val_losses = history.get("val_losses", [])
            print(f"📊 Loaded existing loss history ({len(train_losses)} epochs).")
    else:
        train_losses = []
        val_losses = []

    return train_losses, val_losses