import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from src.dataset import RetinalDataset
from src.preprocessing import get_train_transforms, get_test_transforms
from src.data_loader import prepare_data
from src.model_retfound import build_retfound_model, unfreeze_partial

model_name = "retfound_large"


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

    # 🔥 DIRECTLY UNFREEZE (skip phase 1)
    model = unfreeze_partial(model)
    model = model.to(device)

    criterion = nn.L1Loss()

    # 🔥 LOWER LR for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,
        patience=2
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

    def validate():
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).float()

                features = model.forward_features(imgs)
                outputs = model.head(features).squeeze(1)
                outputs = torch.clamp(outputs, 0, 1)

                preds.extend((outputs * 100).cpu().numpy())
                gts.extend(labels.cpu().numpy())

        return mean_absolute_error(gts, preds)

    # ================= TRAIN LOOP =================
    print("\n🔥 Fine-tuning (Direct Start)")

    for epoch in range(15):
        loss = train_epoch()
        mae = validate()

        scheduler.step(mae)

        print(f"Epoch {epoch+1} | Loss: {loss:.4f} | MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"✅ Best model saved (MAE: {best_mae:.2f})")

    # ================= FINAL SAVE =================
    torch.save(model.state_dict(), f"models/{model_name}_final.pth")

    print("\n🎉 Training complete!")
    print(f"Best MAE: {best_mae:.2f}")


if __name__ == "__main__":
    main()
