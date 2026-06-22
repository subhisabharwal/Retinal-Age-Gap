import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from src.dataset import RetinalDataset
from src.preprocessing import get_train_transforms, get_test_transforms
from src.model import build_model, freeze_backbone, unfreeze_model
from src.data_loader import prepare_data
from sklearn.metrics import mean_absolute_error


def main():
    # =========================================
    # 0. CREATE FOLDERS
    # =========================================
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # =========================================
    # 1. DEVICE
    # =========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True

    # =========================================
    # 2. LOAD DATA
    # =========================================
    data_sources = [
        {
            "type": "odir",
            "csv_path": r"C:\RAG 1\Retinal-Age-Gap\Data\ODIR-5K\data.xlsx",
            "image_folder": r"C:\RAG 1\Retinal-Age-Gap\Data\ODIR-5K\Training Images",
            "patient_id_col": "ID",
            "age_col": "Patient Age",
            "left_col": "Left-Fundus",
            "right_col": "Right-Fundus"
        },
        {
            "type": "new",
            "csv_path": r"C:\RAG 1\Retinal-Age-Gap\Data\BRSET\labels_brset.csv",
            "image_folder": r"C:\RAG 1\Retinal-Age-Gap\Data\BRSET\fundus_photos",
            "patient_id_col": "patient_id",
            "age_col": "patient_age",
            "image_col": "image_id"
        }
    ]

    train_df, val_df = prepare_data(data_sources)

    train_dataset = RetinalDataset(train_df, transform=get_train_transforms())
    val_dataset = RetinalDataset(val_df, transform=get_test_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # =========================================
    # 3. MODEL
    # =========================================
    model = build_model()
    model = freeze_backbone(model)
    model = model.to(device)

    # =========================================
    # 4. LOSS & OPTIMIZER
    # =========================================
    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    scaler = torch.amp.GradScaler("cuda")

    # TensorBoard
    writer = SummaryWriter("runs/experiment1")

    # CSV logging
    history = []

    # =========================================
    # 5. TRAIN FUNCTION
    # =========================================
    def train_one_epoch(model, loader):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        return running_loss / len(loader)

    # =========================================
    # 6. VALIDATION FUNCTION
    # =========================================
    def validate(model, loader):
        model.eval()
        running_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()

                with torch.cuda.amp.autocast():
                    outputs = model(images).squeeze(1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        mae = mean_absolute_error(all_labels, all_preds)

        return running_loss / len(loader), mae

    # =========================================
    # 7. TRAINING LOOP (2 PHASES)
    # =========================================
    best_loss = float("inf")
    epoch_counter = 0

    # ---------- Phase 1 ----------
    print("\n🚀 Phase 1: Frozen Backbone")
    for epoch in range(5):
        epoch_counter += 1
        print(f"\nEpoch {epoch_counter}")

        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_mae = validate(model, val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")

        writer.add_scalar("Loss/train", train_loss, epoch_counter)
        writer.add_scalar("Loss/val", val_loss, epoch_counter)
        writer.add_scalar("MAE/val", val_mae, epoch_counter)

        history.append({
            "epoch": epoch_counter,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")

    # ---------- Phase 2 ----------
    print("\n🔥 Phase 2: Fine-tuning")

    model = unfreeze_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    for epoch in range(20):
        epoch_counter += 1
        print(f"\nEpoch {epoch_counter}")

        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_mae = validate(model, val_loader)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f}")

        writer.add_scalar("Loss/train", train_loss, epoch_counter)
        writer.add_scalar("Loss/val", val_loss, epoch_counter)
        writer.add_scalar("MAE/val", val_mae, epoch_counter)

        history.append({
            "epoch": epoch_counter,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")

    # =========================================
    # 8. SAVE FINAL MODEL
    # =========================================
    torch.save(model.state_dict(), "models/final_model.pth")

    # =========================================
    # 9. SAVE LOGS
    # =========================================
    df = pd.DataFrame(history)
    df.to_csv("logs/training_log.csv", index=False)

    # =========================================
    # 10. PLOTS
    # =========================================
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("logs/loss_curve.png")
    plt.show()

    plt.plot(df["epoch"], df["val_mae"], label="Val MAE")
    plt.legend()
    plt.title("MAE Curve")
    plt.savefig("logs/mae_curve.png")
    plt.show()

    writer.close()

    print("🎉 Training complete!")
    print("📂 Models saved in: models/")
    print("📊 Logs saved in: logs/")


if __name__ == "__main__":
    main()
