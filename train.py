import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import ViolenceDetector
from dataset import ViolenceVideoDataset
from utils import extract_epoch_num, save_loss_plot, save_losses, load_losses, evaluate_and_save, \
    cleanup_old_folders

from torch.optim.lr_scheduler import ReduceLROnPlateau

def main():
    # === System Info ===
    print("CUDA available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


    # === Training Config ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # === Data Loading ===
    # data augmentation to training transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    # simple (just resize + normalize)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    train_dataset = ViolenceVideoDataset(f"{DATA_PATH}/train", transform=train_transform)
    val_dataset = ViolenceVideoDataset(f"{DATA_PATH}/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    # === Model Setup ===
    model = ViolenceDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scaler = GradScaler(device='cuda')


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # mode='min': you want to minimize validation loss.
    #
    # factor=0.5: reduces learning rate by half.
    #
    # patience=5: waits 5 epochs without improvement before reducing LR.

    start_epoch = 0

    # Filter out best model
    checkpoint_files = []
    for root, _, files in os.walk(CHECKPOINT_DIR):
        for file in files:
            if file.endswith('.pt') and 'best' not in file:
                full_path = os.path.join(root, file)
                checkpoint_files.append(full_path)

    # Sort by epoch number
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch_num)

    if checkpoint_files:
        last_ckpt = checkpoint_files[-1]
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

    # Load the best loss from history if it exists
    if val_losses:
        best_loss = min(val_losses)
        print(f"📈 Loaded best validation loss from history: {best_loss:.4f}")
    else:
        best_loss = float('inf')
        print("📈 No validation loss history found. Starting with best_loss = inf")

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

                # Unscale gradients before checking
                scaler.unscale_(optimizer)

                # 🔍 Print gradient norms for debugging
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm():.4f}")

                print("------")
                print(f"attn_query grad norm = {model.attn_query.grad.norm():.6f}")
                print(f"attn_query value norm = {model.attn_query.data.norm():.6f}")

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

            # 🔁 Learning rate scheduler
            scheduler.step(avg_val_loss)

            timestamp = time.strftime("%Y%m%d-%H%M%S")

            folder = os.path.join(CHECKPOINT_DIR, f"violence_detector_epoch{epoch + 1}")
            os.makedirs(folder, exist_ok=True)


            # Save model checkpoint
            ckpt_path = os.path.join(folder, f"violence_detector_epoch{epoch + 1}_{timestamp}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict()
            }, ckpt_path)
            print(f"✅ Saved: {ckpt_path}")


            # Keep only latest N checkpoints Cleanup old folders
            cleanup_old_folders(CHECKPOINT_DIR, MAX_CHECKPOINTS)

            # Save loss history
            save_losses(train_losses, val_losses, loss_log_path=loss_log_path)

            # === Plot Training Curve ===
            save_loss_plot(train_losses, val_losses, loss_curve_path)



            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_folder = os.path.join(CHECKPOINT_DIR, "violence_detector_best")
                os.makedirs(best_folder, exist_ok=True)

                # Save the model weights
                best_model_path = os.path.join(best_folder, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)

                # Save evaluation
                evaluate_and_save(epoch, model, val_loader, device, best_folder)

                print(f"🏆 Best model updated and saved at: {best_folder}")
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

    EPOCHS = 200  # Reduce for testing
    # ✅ Hyperparameters
    SEQ_LEN = 20
    BATCH_SIZE = 12
    LR = 1e-5
    MAX_CHECKPOINTS = 5
    PATIENCE = 50

    DATA_PATH = "/home/soufianejd/datasets/violence"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    loss_log_path = os.path.join(CHECKPOINT_DIR, "loss_history.json")
    loss_curve_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")

    main()