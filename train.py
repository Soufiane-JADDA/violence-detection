import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from model import ViolenceDetector
from dataset import ViolenceVideoDataset
from utils import extract_epoch_num, save_loss_plot, save_losses, load_losses

def main():
    # === System Info ===
    print("CUDA available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


    # === Training Config ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # === Data Loading ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])


    dataset = ViolenceVideoDataset(DATA_PATH, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    # === Model Setup ===
    model = ViolenceDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(device='cuda')

    start_epoch = 0

    # Filter out best model
    checkpoint_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith('.pt') and f != 'violence_detector_best.pt'
    ]

    # Sort by epoch number
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch_num)

    if checkpoint_files:
        last_ckpt = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
        print(f"🔄 Loading checkpoint: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            start_epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)



    # Load existing loss data if available
    train_losses, val_losses = load_losses(loss_log_path)

    best_loss = float('inf')
    patience_counter = 0

    # === Training Loop ===
    model.train()
    try:

        for epoch in range(start_epoch, EPOCHS):
            total_loss = 0.0
            model.train()
            loop = tqdm(train_loader, desc=f"🟢 Epoch {epoch + 1}/{EPOCHS}", leave=False)
            for clips, targets in loop:
                clips = clips.to(device)
                targets = targets.clone().detach().to(device)

                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = model(clips)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for clips, targets in val_loader:
                    clips = clips.to(device)
                    targets = targets.to(device)
                    with autocast(device_type='cuda'):
                        outputs = model(clips)
                        loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"📘 Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"violence_detector_epoch{epoch+1}_{timestamp}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict()
            }, ckpt_path)
            print(f"✅ Saved: {ckpt_path}")

            # Keep only latest N checkpoints
            checkpoints = sorted([
                f for f in os.listdir(CHECKPOINT_DIR)
                if f.endswith('.pt') and f != 'violence_detector_best.pt' and f != os.path.basename(ckpt_path)
            ], key=extract_epoch_num)

            if len(checkpoints) > MAX_CHECKPOINTS:
                to_remove = checkpoints[0]
                os.remove(os.path.join(CHECKPOINT_DIR, to_remove))
                print(f"🧹 Removed oldest checkpoint: {to_remove}")


            # Save loss history
            save_losses(train_losses, val_losses, loss_log_path="loss_history.json")

            # === Plot Training Curve ===
            save_loss_plot(train_losses, val_losses, loss_curve_path)

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_path = os.path.join(CHECKPOINT_DIR, "violence_detector_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"🏆 Best model updated: {best_path}")
                patience_counter = 0
            else:
                # Early Stopping: If validation loss doesn’t improve for PATIENCE consecutive epochs, training stops early.
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("⏹️ Early stopping: No improvement in validation loss.")
                    break

    except KeyboardInterrupt:
        print("🛑 Training interrupted by user.")


if __name__=="__main__":

    EPOCHS = 1000  # Reduce for testing
    # ✅ Hyperparameters
    SEQ_LEN = 20
    BATCH_SIZE = 2
    LR = 0.0001
    MAX_CHECKPOINTS = 5
    PATIENCE = 50

    DATA_PATH = "/mnt/SDrive/temp 2/real-life-violence-situations-dataset/Real Life Violence Dataset"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    loss_log_path = os.path.join(CHECKPOINT_DIR, "loss_history.json")
    loss_curve_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")

    main()