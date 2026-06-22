import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from src.dataset import RetinalDataset
from src.preprocessing import get_test_transforms
from src.data_loader import prepare_data
from src.model import build_model  # your old EfficientNet
from src.model_retfound import build_retfound_model


def main():
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

    _, val_df = prepare_data(data_sources)

    val_dataset = RetinalDataset(val_df, transform=get_test_transforms())
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # ================= LOAD MODELS =================

    # EfficientNet (OLD working model)
    model_eff = build_model()
    model_eff.load_state_dict(torch.load("models/best_model.pth"))
    model_eff = model_eff.to(device)
    model_eff.eval()

    # RETFound
    model_ret = build_retfound_model("pretrained/retfound_large.pth")
    model_ret.load_state_dict(torch.load("models/retfound_large_best.pth"))
    model_ret = model_ret.to(device)
    model_ret.eval()

    preds, gts = [], []

    # ================= INFERENCE =================
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs = imgs.to(device)
            labels = labels.to(device).float()   # IMPORTANT: keep labels in original scale

            # -------- EfficientNet --------
            pred_eff = model_eff(imgs).squeeze(1)

            # -------- RETFound --------
            feat = model_ret.forward_features(imgs)
            pred_ret = model_ret.head(feat).squeeze(1)

            # -------- FIX: ensure same scale --------
            pred_eff = torch.clamp(pred_eff, 0, 1)
            pred_ret = torch.clamp(pred_ret, 0, 1)

            # -------- ENSEMBLE --------
            final_pred = 0.7 * pred_ret + 0.3 * pred_eff

            # -------- Convert back to age --------
            final_pred = final_pred * 100

            preds.extend(final_pred.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    mae = mean_absolute_error(gts, preds)

    print("\n==============================")
    print(f"🔥 FINAL ENSEMBLE MAE: {mae:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()
