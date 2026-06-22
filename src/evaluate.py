import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

from src.dataset import RetinalDataset
from src.preprocessing import get_test_transforms
from src.model import build_model
from src.data_loader import prepare_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SAME DATA CONFIG
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

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # LOAD MODEL
    model = build_model()
    model.load_state_dict(torch.load("models/best_model.pth"))
    model = model.to(device)
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for images, y in val_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)

            preds.extend(outputs.cpu().numpy())
            labels.extend(y.numpy())

    # CALCULATE MAE
    mae = mean_absolute_error(labels, preds)
    print(f"Final MAE: {mae:.2f}")

    # SAVE RESULTS
    df = pd.DataFrame({
        "Actual": labels,
        "Predicted": preds
    })
    df.to_csv("logs/predictions.csv", index=False)

    # PLOT
    plt.scatter(labels, preds, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Predictions vs Actual")
    plt.savefig("logs/prediction_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
