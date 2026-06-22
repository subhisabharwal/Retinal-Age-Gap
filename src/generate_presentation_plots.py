import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

from src.dataset import RetinalDataset
from src.preprocessing import get_test_transforms
from src.data_loader import prepare_data

# -------------------------------------------------
# IMPORT MODELS
# -------------------------------------------------

from src.model import build_model as build_effnet_b0

from src.model_efficient_b3 import (
    build_model as build_effnet_b3
)

from src.model_retfound import (
    build_retfound_model
)

# =================================================
# MAIN
# =================================================

def main():

    # -------------------------------------------------
    # DEVICE
    # -------------------------------------------------

    device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    # -------------------------------------------------
    # OUTPUT FOLDER
    # -------------------------------------------------

    os.makedirs(
        "outputs/presentation",
        exist_ok=True
    )

    # =================================================
    # DATA SOURCES
    # =================================================

    data_sources = [

        {
            "type": "odir",

            "csv_path":
            "data/ODIR-5K/data.xlsx",

            "image_folder":
            "data/ODIR-5K/Training Images",

            "patient_id_col": "ID",

            "age_col": "Patient Age",

            "left_col": "Left-Fundus",

            "right_col": "Right-Fundus"
        },

        {
            "type": "new",

            "csv_path":
            "data/BRSET/labels_brset.csv",

            "image_folder":
            "data/BRSET/fundus_photos",

            "patient_id_col": "patient_id",

            "age_col": "patient_age",

            "image_col": "image_id"
        }
    ]

    # =================================================
    # LOAD DATA
    # =================================================

    print("\nPreparing validation data...")

    _, val_df = prepare_data(data_sources)

    # -------------------------------------------------
    # FAST PRESENTATION SUBSET
    # -------------------------------------------------

    val_df = val_df.sample(
        200,
        random_state=42
    )

    print(
        f"Using {len(val_df)} validation samples"
    )

    # -------------------------------------------------

    val_dataset = RetinalDataset(
        val_df,
        transform=get_test_transforms()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # =================================================
    # LOAD MODELS
    # =================================================

    print("\nLoading models...")

    # -------------------------------------------------
    # EfficientNet-B0
    # -------------------------------------------------

    model1 = build_effnet_b0()

    model1.load_state_dict(
        torch.load(
            "model/models/best_model.pth",
            map_location=device
        )
    )

    model1 = model1.to(device)

    model1.eval()

    print("✅ Loaded EfficientNet-B0")

    # -------------------------------------------------
    # EfficientNet-B3
    # -------------------------------------------------

    model2 = build_effnet_b3()

    model2.load_state_dict(
        torch.load(
            "model/models/effnet_b3_best.pth",
            map_location=device
        )
    )

    model2 = model2.to(device)

    model2.eval()

    print("✅ Loaded EfficientNet-B3")

    # -------------------------------------------------
    # RETFound Linear Probe
    # -------------------------------------------------

    model3 = build_retfound_model(
        "pretrained/retfound_large.pth"
    )

    model3.load_state_dict(
        torch.load(
            "model/models/retfound_large_best.pth",
            map_location=device
        )
    )

    model3 = model3.to(device)

    model3.eval()

    print("✅ Loaded RETFound Linear Probe")

    # -------------------------------------------------
    # RETFound Final Fine-Tuned
    # -------------------------------------------------

    model4 = build_retfound_model(
        "pretrained/retfound_large.pth"
    )

    model4.load_state_dict(
        torch.load(
            "model/models/retfound_final_best.pth",
            map_location=device
        )
    )

    model4 = model4.to(device)

    model4.eval()

    print("✅ Loaded RETFound Final")

    # =================================================
    # INFERENCE
    # =================================================

    preds1 = []
    preds2 = []
    preds3 = []
    preds4 = []

    gts = []

    print("\nRunning inference...")

    with torch.no_grad():

        for imgs, labels in tqdm(val_loader):

            imgs = imgs.to(device)

            labels = labels.float()

            # =========================================
            # EfficientNet-B0
            # =========================================

            p1 = model1(imgs)

            p1 = p1.squeeze(1).cpu()

            # =========================================
            # EfficientNet-B3
            # =========================================

            p2 = model2(imgs)

            p2 = p2.squeeze(1).cpu()

            # =========================================
            # RETFound Linear Probe
            # =========================================

            feat3 = model3.forward_features(imgs)

            p3 = model3.head(feat3)

            p3 = p3.squeeze(1).cpu()

            # =========================================
            # RETFound Final
            # =========================================

            feat4 = model4.forward_features(imgs)

            p4 = model4.head(feat4)

            p4 = p4.squeeze(1).cpu()

            # =========================================
            # STORE PREDICTIONS
            # =========================================

            preds1.extend(
                (
                    p1 * 100
                    if p1.mean() < 5
                    else p1
                ).numpy()
            )

            preds2.extend(
                (
                    p2 * 100
                    if p2.mean() < 5
                    else p2
                ).numpy()
            )

            preds3.extend(
                (
                    p3 * 100
                    if p3.mean() < 5
                    else p3
                ).numpy()
            )

            preds4.extend(
                (
                    p4 * 100
                    if p4.mean() < 5
                    else p4
                ).numpy()
            )

            gts.extend(labels.numpy())

    # =================================================
    # CONVERT TO NUMPY
    # =================================================

    gts = np.array(gts)

    preds1 = np.array(preds1)

    preds2 = np.array(preds2)

    preds3 = np.array(preds3)

    preds4 = np.array(preds4)

    # =================================================
    # MODEL RESULTS
    # =================================================

    model_results = {

        "EfficientNet-B0": {
            "preds": preds1
        },

        "EfficientNet-B3": {
            "preds": preds2
        },

        "RETFound_Linear_Probe": {
            "preds": preds3
        },

        "RETFound_Final": {
            "preds": preds4
        }
    }

    # =================================================
    # COMPUTE METRICS
    # =================================================

    for name in model_results:

        preds = model_results[name]["preds"]

        mae = mean_absolute_error(
            gts,
            preds
        )

        r2 = r2_score(
            gts,
            preds
        )

        model_results[name]["mae"] = mae

        model_results[name]["r2"] = r2

    # =================================================
    # SAVE METRICS TABLE
    # =================================================

    metrics_df = pd.DataFrame({

        "Model":
        list(model_results.keys()),

        "MAE":
        [
            model_results[m]["mae"]
            for m in model_results
        ],

        "R2":
        [
            model_results[m]["r2"]
            for m in model_results
        ]
    })

    metrics_df.to_csv(
        "outputs/presentation/model_metrics.csv",
        index=False
    )

    print("\n✅ Saved metrics table")

    # =================================================
    # MAE COMPARISON
    # =================================================

    plt.figure(figsize=(8, 6))

    plt.bar(
        metrics_df["Model"],
        metrics_df["MAE"]
    )

    plt.ylabel("Mean Absolute Error")

    plt.title(
        "Model MAE Comparison"
    )

    plt.grid(
        axis="y",
        linestyle=":"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/presentation/mae_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("✅ Saved MAE comparison")

    # =================================================
    # R² COMPARISON
    # =================================================

    plt.figure(figsize=(8, 6))

    plt.bar(
        metrics_df["Model"],
        metrics_df["R2"]
    )

    plt.ylabel("R² Score")

    plt.title(
        "Model R² Comparison"
    )

    plt.grid(
        axis="y",
        linestyle=":"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/presentation/r2_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("✅ Saved R² comparison")

    # =================================================
    # INDIVIDUAL MODEL PLOTS
    # =================================================

    for name, info in model_results.items():

        preds = info["preds"]

        mae = info["mae"]

        r2 = info["r2"]

        # =============================================
        # SCATTER PLOT
        # =============================================

        plt.figure(figsize=(7, 6))

        plt.scatter(
            gts,
            preds,
            alpha=0.5,
            s=25
        )

        min_val = min(
            gts.min(),
            preds.min()
        )

        max_val = max(
            gts.max(),
            preds.max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--"
        )

        plt.xlabel("Actual Age")

        plt.ylabel("Predicted Age")

        plt.title(
            f"{name}\nPredicted vs Actual Age"
        )

        metrics_text = (
            f"MAE: {mae:.2f}\n"
            f"R²: {r2:.2f}"
        )

        plt.text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8
            )
        )

        plt.grid(True)

        plt.tight_layout()

        scatter_path = (
            f"outputs/presentation/"
            f"{name}_scatter.png"
        )

        plt.savefig(
            scatter_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"✅ Saved: {scatter_path}"
        )

        # =============================================
        # ERROR DISTRIBUTION
        # =============================================

        errors = preds - gts

        plt.figure(figsize=(7, 6))

        plt.hist(
            errors,
            bins=20,
            alpha=0.7
        )

        plt.xlabel("Prediction Error")

        plt.ylabel("Frequency")

        plt.title(
            f"{name}\nPrediction Error Distribution"
        )

        plt.grid(True)

        plt.tight_layout()

        error_path = (
            f"outputs/presentation/"
            f"{name}_error_distribution.png"
        )

        plt.savefig(
            error_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"✅ Saved: {error_path}"
        )

    # =================================================
    # DONE
    # =================================================

    print(
        "\n🎉 ALL PRESENTATION PLOTS GENERATED"
    )

    print(
        "\nSaved inside:"
    )

    print(
        "outputs/presentation/"
    )

# =================================================
# RUN
# =================================================

if __name__ == "__main__":

    main()
