import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_video_dataset(input_dir, output_dir, split_ratio=0.8, seed=42):
    random.seed(seed)
    classes = ['Fight', 'NonFight']

    for cls in classes:
        print(f"\n📂 Processing class: {cls}")
        class_dir = Path(input_dir) / cls
        videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        random.shuffle(videos)

        split_idx = int(len(videos) * split_ratio)
        train_videos = videos[:split_idx]
        val_videos = videos[split_idx:]

        for subset, subset_videos in [('train', train_videos), ('val', val_videos)]:
            subset_dir = Path(output_dir) / subset / cls
            subset_dir.mkdir(parents=True, exist_ok=True)

            print(f"➡️ Copying {len(subset_videos)} videos to {subset_dir}")
            for vid_path in tqdm(subset_videos, desc=f"Copying to {subset}/{cls}"):
                shutil.copy(vid_path, subset_dir / vid_path.name)

    print(f"\n✅ Dataset split complete. Output saved in: {output_dir}")

split_video_dataset(
    input_dir="/mnt/SDrive/temp/RWF-2000/train",
    output_dir="/mnt/SDrive/temp/datasetFight",
    split_ratio=0.8
)
