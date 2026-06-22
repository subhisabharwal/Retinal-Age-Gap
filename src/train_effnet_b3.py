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
from src.model_efficient_b3 import build_model, freeze_backbone, unfreeze_partial
from src.data_loader import prepare_data
from sklearn.metrics import mean_absolute_error

model_name = "effnet_b3"


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True

    # ================= DATA =================
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
        batch_size=32,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    # ================= MODEL =================
    model = build_model()
    model = freeze_backbone(model)
    model = model.to(device)

    # ================= LOSS =================
    criterion = nn.SmoothL1Loss()

    # ================= PHASE 1 =================
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    writer = SummaryWriter(f"runs/{model_name}")
    history = []

    # ================= TRAIN =================
    def train_one_epoch(model, loader):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device).float()

            # 🔥 NORMALIZE LABELS
            labels = labels / 100.0

            optimizer.zero_grad()

            outputs = model(images).squeeze(1)

            # 🔥 STABLE RANGE
            outputs = torch.clamp(outputs, 0, 1)

            loss = criterion(outputs, labels)

            loss.backward()

            # 🔥 gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    # ================= VALIDATE =================
    def validate(model, loader):
        model.eval()
        running_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device).float()

                labels_norm = labels / 100.0

                outputs = model(images).squeeze(1)
                outputs = torch.clamp(outputs, 0, 1)

                loss = criterion(outputs, labels_norm)
                running_loss += loss.item()

                # 🔥 denormalize predictions
                preds = outputs * 100.0

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        mae = mean_absolute_error(all_labels, all_preds)

        return running_loss / len(loader), mae

    # ================= TRAIN LOOP =================
    best_loss = float("inf")
    epoch_counter = 0

    print("\n🚀 Phase 1: Frozen Backbone")

    for epoch in range(5):
        epoch_counter += 1
        print(f"\nEpoch {epoch_counter}")

        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_mae = validate(model, val_loader)

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
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")

    # ================= PHASE 2 =================
    print("\n🔥 Phase 2: Fine-tuning")

    model = unfreeze_partial(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    for epoch in range(15):
        epoch_counter += 1
        print(f"\nEpoch {epoch_counter}")

        train_loss = train_one_epoch(model, train_loader)
        val_loss, val_mae = validate(model, val_loader)

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
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")

    # ================= SAVE =================
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")

    df = pd.DataFrame(history)
    df.to_csv(f"logs/{model_name}_training_log.csv", index=False)

    writer.close()

    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
