import os
import torch
from torch.utils.data import Dataset
from utils import extract_frames

# === Dataset ===
class ViolenceVideoDataset(Dataset):
    def __init__(self, video_dir, seq_len=20, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.data = []
        self.labels = []

        for label, folder in enumerate(["NonViolence", "Violence"]):
            path = os.path.join(video_dir, folder)
            for video in os.listdir(path):
                video_path = os.path.join(path, video)
                self.data.append(video_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        frames = extract_frames(video_path, max_frames=self.seq_len)
        if len(frames) < self.seq_len:
            frames = [frames[0]] * self.seq_len  # Pad

        if self.transform:
            frames = [self.transform(f).unsqueeze(0) for f in frames]

        clip = torch.cat(frames, dim=0)  # (T, C, H, W)
        return clip, self.labels[idx]