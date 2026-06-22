import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from src.dataset import RetinalDataset
from src.preprocessing import get_train_transforms, get_test_transforms
from src.data_loader import prepare_data
from src.model_retfound import build_retfound_model

model_name = "retfound_final"


def main():
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # ================= MODEL =================
    model = build_retfound_model("pretrained/retfound_large.pth")

    # 🔥 PARTIAL UNFREEZE (last blocks only)
    for name, param in model.named_parameters():
        if "blocks.10" in name or "blocks.11" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)

    # ================= LOSS =================
    criterion = nn.SmoothL1Loss()

    # ================= OPTIMIZER =================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-6,
        weight_decay=5e-4
    )

    # ================= ADAPTIVE LR =================
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=1
    )

    best_mae = float("inf")

    # ================= TRAIN =================
    def train_epoch():
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device).float() / 100.0

            optimizer.zero_grad()

            features = model.forward_features(imgs)
            outputs = model.head(features).squeeze(1)
            outputs = torch.clamp(outputs, 0, 1)

            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    # ================= VALIDATION =================
    def validate(use_tta=False):
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float()

                # -------- NORMAL PRED --------
                feat = model.forward_features(imgs)
                pred = model.head(feat).squeeze(1)

                if use_tta:
                    # -------- TTA (horizontal flip) --------
                    imgs_flip = torch.flip(imgs, dims=[3])
                    feat_flip = model.forward_features(imgs_flip)
                    pred_flip = model.head(feat_flip).squeeze(1)

                    pred = (pred + pred_flip) / 2

                pred = torch.clamp(pred, 0, 1)

                preds.extend((pred * 100).cpu().numpy())
                gts.extend(labels.cpu().numpy())

        return mean_absolute_error(gts, preds)

    # ================= TRAIN LOOP =================
    print("\n🔥 Final Optimized Training")

    for epoch in range(8):
        loss = train_epoch()
        mae = validate(use_tta=False)

        scheduler.step(mae)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"✅ Best model saved (MAE: {best_mae:.2f})")

    # ================= FINAL EVALUATION WITH TTA =================
    print("\n🚀 Evaluating with TTA...")
    final_mae = validate(use_tta=True)

    print("\n==============================")
    print(f"🔥 FINAL TTA MAE: {final_mae:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()
