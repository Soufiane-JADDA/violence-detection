import os
import torch
from model import ViolenceDetector
from utils import extract_frames
from torchvision import transforms
import torch.nn.functional as functional
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoints/violence_detector_epoch123_20250517-185702.pt"
data_dir = r"/home/soufianejd/Downloads/data"
seq_len = 20  # subset of your dataset

# Load model
model = ViolenceDetector().to(device)

model_file = torch.load(model_path, map_location=device)
if isinstance(model_file, dict) and 'model_state' in model_file:
    model.load_state_dict(model_file['model_state'])
else:
    model.load_state_dict(model_file)

model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Data collection
y_true = []
y_pred = []
misclassified = []

# Evaluate all videos
for label, folder in enumerate(["NonViolence", "Violence"]):
    folder_path = os.path.join(data_dir, folder)
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        frames = extract_frames(path, max_frames=seq_len)
        if len(frames) < seq_len:
            continue

        frames = [transform(f).unsqueeze(0) for f in frames]
        clip = torch.cat(frames, dim=0).unsqueeze(0).to(device)  # (1, T, C, H, W)

        with torch.no_grad():
            output = model(clip)
            probs = functional.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()

        y_true.append(label)
        y_pred.append(pred)

        if pred != label:
            misclassified.append((file, "Violent" if label == 1 else "Non-Violent", "Violent" if pred == 1 else "Non-Violent"))

# Show classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Non-Violent", "Violent"]))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Violent", "Violent"], yticklabels=["Non-Violent", "Violent"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save misclassified results
with open("misclassified_samples.txt", "w") as f:
    for filename, actual, predicted in misclassified:
        f.write(f"{filename}: Actual = {actual}, Predicted = {predicted}\n")
