import os
import cv2
import re
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shutil
import matplotlib
matplotlib.use("Agg")  # ✅ Use non-interactive backend


# Frame Extraction with better sampling and corrupted video handling
def extract_frames(video_path, max_frames=20, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0 or not cap.isOpened():
        print(f"⚠️ Skipping unreadable video: {video_path}")
        dummy = np.zeros((*resize, 3), dtype=np.uint8)
        return [dummy] * max_frames

    frame_ids = np.linspace(0, total - 1, max_frames).astype(int)
    current_id = 0
    count = 0

    while count < total and len(frames) < max_frames:
        ret, frame = cap.read()
        if count == frame_ids[current_id]:
            if not ret or frame is None:
                print(f"⚠️ Error reading frame {count} in {video_path}")
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((*resize, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, resize)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)
            current_id += 1
            if current_id >= len(frame_ids):
                break
        count += 1

    cap.release()

    # Pad if needed
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())

    return frames



# Extract epoch number from checkpoint filename
def extract_epoch_num(filename):
    match = re.search(r'epoch(\d+)', filename)
    return int(match.group(1)) if match else -1


# Save training and validation loss plot
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


# Save loss history to JSON file
def save_losses(train_losses, val_losses, loss_log_path="loss_history.json"):
    with open(loss_log_path, "w") as file:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, file)


# Load loss history from JSON file
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



def save_confusion_matrix(y_true, y_pred, output_path, class_names=("NonFight", "Fight")):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Acc={acc:.2%})')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def evaluate_and_save(epoch, model, dataloader, device, folder):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for clips, targets in dataloader:
            clips, targets = clips.to(device), targets.to(device)
            outputs = model(clips)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    if not y_true or not y_pred:
        print("⚠️ Warning: No predictions were made.")
        return 0.0

    cm = confusion_matrix(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["NonFight", "Fight"])

    os.makedirs(folder, exist_ok=True)

    # Save extended accuracy report
    with open(os.path.join(folder, "accuracy.txt"), "w") as f:
        f.write(f"Epoch: {epoch+1}\n")
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    save_confusion_matrix(y_true, y_pred, os.path.join(folder, "confusion_matrix.png"))
    return acc



def cleanup_old_folders(base_dir, max_folders):
    folders = [f for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("violence_detector_epoch")]

    folders = sorted(folders, key=lambda x: int(x.replace("violence_detector_epoch", "").split("_")[0]))

    while len(folders) > max_folders:
        to_remove = folders.pop(0)
        path = os.path.join(base_dir, to_remove)
        if 'best' not in path:  # Do not remove best model folder
            shutil.rmtree(path)
            print(f"🗑️ Removed old checkpoint folder: {path}")

