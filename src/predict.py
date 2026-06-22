import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

from src.dataset import RetinalDataset
from src.preprocessing import get_test_transforms
from src.model_retfound import build_retfound_model
from src.data_loader import prepare_data


def load_state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Invalid checkpoint format")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "")
        cleaned_state_dict[key] = value

    return cleaned_state_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("outputs", exist_ok=True)

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

    print("First 5 validation image paths:")
    for path in val_df["image_path"].head(5).tolist():
        print(" ", path)

    val_dataset = RetinalDataset(val_df, transform=get_test_transforms())
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Validation samples: {len(val_df)}")

    model = build_retfound_model("pretrained/retfound_large.pth")

    checkpoint = torch.load("models/retfound_large_best.pth", map_location=device)
    state_dict = load_state_dict_from_checkpoint(checkpoint)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print("✅ Model loaded from models/retfound_large_best.pth")

    results = []
    predicted_ages = []
    true_ages = []

    print("\nRunning predictions...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            features = model.forward_features(images)
            outputs = model.head(features).squeeze(1)
            outputs = torch.clamp(outputs, 0.0, 1.0)
            outputs = outputs * 100.0

            predicted_ages.extend(outputs.cpu().numpy())
            true_ages.extend(labels.numpy())

    for idx, (pred_age, true_age) in enumerate(zip(predicted_ages, true_ages)):
        image_path = val_df.iloc[idx]["image_path"]
        results.append({
            "image_path": image_path,
            "predicted_age": float(pred_age),
            "true_age": float(true_age),
            "RAG": float(pred_age - true_age)
        })

    results_df = pd.DataFrame(results)
    output_path = "outputs/predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved at: {output_path}")

    mae = mean_absolute_error(true_ages, predicted_ages)
    mean_rag = results_df["RAG"].mean()
    std_rag = results_df["RAG"].std()

    print(f"\n📊 METRICS:")
    print(f"   Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"   Mean RAG (Retinal Age Gap): {mean_rag:.2f} years")
    print(f"   Std RAG: {std_rag:.2f} years")
    print(f"   Total predictions: {len(results_df)}")

    metrics_summary = {
        "metric": ["MAE", "Mean_RAG", "Std_RAG", "Total_Samples"],
        "value": [mae, mean_rag, std_rag, len(results_df)]
    }

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_path = "outputs/metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Metrics summary saved at: {metrics_path}")


if __name__ == "__main__":
    main()
