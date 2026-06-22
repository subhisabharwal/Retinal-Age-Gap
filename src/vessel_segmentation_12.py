import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi

INPUT_CSV = "outputs/predictions.csv"
OUTPUT_DIR = "outputs/vessel_masks_12"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 512

print("🚀 Balanced vessel segmentation...")


def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    green = image[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    return enhanced


df = pd.read_csv(INPUT_CSV)

for i, row in df.iterrows():

    img_path = row["image_path"]

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)

    processed = preprocess(image)

    # 🔥 FRANGI (key)
    vessels = frangi(processed, scale_range=(1, 3), scale_step=1)

    vessels = (vessels - vessels.min()) / (vessels.max() + 1e-8)

    # 🔥 BALANCED threshold
    thresh = np.percentile(vessels, 85)
    vessel_mask = (vessels > thresh).astype(np.uint8) * 255

    # 🔥 VERY LIGHT cleanup (only tiny noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask, connectivity=8)

    clean_mask = np.zeros_like(vessel_mask)

    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]

        if area > 15:   # only remove tiny dots
            clean_mask[labels == j] = 255

    vessel_mask = clean_mask

    # 🔥 NO skeletonization (important)
    # 🔥 NO erosion (important)

    # ----------------------------
    # SAVE
    # ----------------------------
    base = os.path.splitext(os.path.basename(img_path))[0]

    cv2.imwrite(os.path.join(OUTPUT_DIR, base + "_vessel.png"), vessel_mask)

    overlay = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    overlay[vessel_mask == 255] = [0, 0, 255]

    cv2.imwrite(os.path.join(OUTPUT_DIR, base + "_overlay.png"), overlay)

    if i % 50 == 0:
        print(f"Processed {i}")

print("✅ DONE")


if __name__ == "__main__":
    print("Starting...")
