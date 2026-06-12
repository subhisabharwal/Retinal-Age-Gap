import os
import subprocess
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import cv2
import sys

# =====================================================
# CONFIG
# =====================================================

IMAGE_PATH = "data/BRSET/fundus_photos/img00001.jpg"

OUTPUT_DIR = "outputs/demo_case"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# STEP 1 — RETFOUND PREDICTION
# =====================================================

print("\n🚀 Running RETFound prediction...")

subprocess.run([
    sys.executable,
    "-m",
    "src.predict",
    IMAGE_PATH
], check=True)

print("✅ RETFound prediction complete")

# =====================================================
# STEP 2 — VESSEL SEGMENTATION
# =====================================================

print("\n🚀 Running vessel segmentation...")

subprocess.run([
    "python",
    "-m",
    "src.vessel_segmentation_final",
    IMAGE_PATH
])

print("✅ Vessel segmentation complete")

# =====================================================
# STEP 3 — FEATURE EXTRACTION
# =====================================================

print("\n🚀 Extracting vascular biomarkers...")

subprocess.run([
    "python",
    "-m",
    "src.extract_vascular_feature"
])

print("✅ Biomarker extraction complete")

# =====================================================
# STEP 4 — LOCAL SHAP ANALYSIS
# =====================================================

print("\n🚀 Running local SHAP explanation...")

subprocess.run([
    "python",
    "-m",
    "src.local_shap"
])

print("✅ Local SHAP complete")

# =====================================================
# STEP 5 — LOAD RESULTS
# =====================================================

print("\n🚀 Loading generated outputs...")

feature_df = pd.read_csv(
    "outputs/biomarker_analysis.csv"
)

print(feature_df)

# =====================================================
# STEP 6 — FINAL SUMMARY PANEL
# =====================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

# -----------------------------------------------------
# ORIGINAL IMAGE
# -----------------------------------------------------

original = cv2.imread(IMAGE_PATH)

original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

axes[0].imshow(original)

axes[0].set_title(
    "Original Retinal Image"
)

axes[0].axis("off")

# -----------------------------------------------------
# SEGMENTATION
# -----------------------------------------------------

segmentation = cv2.imread(
    "outputs/vessel_segmentation.png",
    0
)

axes[1].imshow(
    segmentation,
    cmap="gray"
)

axes[1].set_title(
    "Vessel Segmentation"
)

axes[1].axis("off")

# -----------------------------------------------------
# LOCAL SHAP IMAGE
# -----------------------------------------------------

shap_plot = cv2.imread(
    "outputs/local_feature_explanation.png"
)

shap_plot = cv2.cvtColor(
    shap_plot,
    cv2.COLOR_BGR2RGB
)

axes[2].imshow(shap_plot)

axes[2].set_title(
    "Local Biomarker Explanation"
)

axes[2].axis("off")

# -----------------------------------------------------

plt.tight_layout()

summary_path = os.path.join(
    OUTPUT_DIR,
    "final_summary_panel.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

print("✅ Saved final summary panel")

# =====================================================
# DONE
# =====================================================

print(
    "\n🎉 FULL PIPELINE COMPLETE"
)

print(
    "\nGenerated:"
)

print("- RETFound prediction")
print("- Vessel segmentation")
print("- Biomarker extraction")
print("- Local SHAP explanation")
print("- Medical interpretation")
print("- Final summary panel")
