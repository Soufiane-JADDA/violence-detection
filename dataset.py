import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
import numpy as np

class ViolenceVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, seq_len=20):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.samples = []

        classes = ['NonFight', 'Fight']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for label in classes:
            label_dir = os.path.join(self.root_dir, label)
            for fname in os.listdir(label_dir):
                if fname.endswith((".mp4", ".avi")):
                    self.samples.append({
                        'path': os.path.join(label_dir, fname),
                        'label': self.class_to_idx[label]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.load_video_frames(sample['path'], self.seq_len)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        video_tensor = torch.stack(frames)  # (T, C, H, W)
        return video_tensor, torch.tensor(sample['label'])

    def load_video_frames(self, video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0 or not cap.isOpened():
            print(f"⚠️ Skipping corrupt or unreadable video: {video_path}")
            cap.release()
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            return [dummy for _ in range(num_frames)]

        if total < num_frames:
            indices = [i % total for i in range(num_frames)]
        else:
            start = random.randint(0, total - num_frames)
            indices = list(range(start, start + num_frames))

        frames = []
        current_frame = 0

        while current_frame < total and len(frames) < num_frames:
            ret, frame = cap.read()
            if current_frame in indices:
                if not ret or frame is None:
                    print(f"⚠️ Error reading frame {current_frame} in {video_path}")
                    frame = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            current_frame += 1

        cap.release()

        while len(frames) < num_frames:
            frames.append(frames[-1])

        return frames
