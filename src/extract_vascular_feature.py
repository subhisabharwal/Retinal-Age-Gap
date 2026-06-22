import os
import cv2
import numpy as np
import pandas as pd

from skimage.morphology import skeletonize
from scipy.ndimage import convolve

# -----------------------------
# PATHS
# -----------------------------
VESSEL_DIR = "outputs/vessel_masks_final"
OUTPUT_CSV = "outputs/vascular_features.csv"

os.makedirs("outputs", exist_ok=True)

results = []

print("🚀 Extracting vascular biomarkers...")

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------

def vessel_density(mask):
    return np.sum(mask > 0) / mask.size


def vessel_length(skel):
    return np.sum(skel)


def branching_points(skel):
    kernel = np.array([
        [1,1,1],
        [1,10,1],
        [1,1,1]
    ])

    neighbors = convolve(skel.astype(np.uint8), kernel, mode='constant')

    # branch point = center + >=3 neighbors
    branches = np.sum(neighbors >= 13)

    return branches


def fractal_dimension(mask):
    mask = mask > 0

    sizes = 2 ** np.arange(1, 8)

    counts = []

    for size in sizes:
        S = np.add.reduceat(
            np.add.reduceat(mask, np.arange(0, mask.shape[0], size), axis=0),
            np.arange(0, mask.shape[1], size), axis=1
        )

        counts.append(np.sum(S > 0))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]


def tortuosity(skel):
    coords = np.column_stack(np.where(skel > 0))

    if len(coords) < 2:
        return 0

    total_path = len(coords)

    start = coords[0]
    end = coords[-1]

    euclidean = np.linalg.norm(start - end)

    if euclidean == 0:
        return 0

    return total_path / euclidean


def average_thickness(mask, skel):
    vessel_area = np.sum(mask > 0)
    skeleton_length = np.sum(skel)

    if skeleton_length == 0:
        return 0

    return vessel_area / skeleton_length


# -----------------------------
# MAIN LOOP
# -----------------------------

for file in os.listdir(VESSEL_DIR):

    if not file.endswith(".png"):
        continue

    path = os.path.join(VESSEL_DIR, file)

    mask = cv2.imread(path, 0)

    if mask is None:
        continue

    # binary
    mask = (mask > 127).astype(np.uint8)

    # skeleton
    skel = skeletonize(mask).astype(np.uint8)

    # -----------------------------
    # FEATURES
    # -----------------------------
    density = vessel_density(mask)

    length = vessel_length(skel)

    branches = branching_points(skel)

    fractal = fractal_dimension(mask)

    tort = tortuosity(skel)

    thickness = average_thickness(mask, skel)

    results.append({
        "image": file,
        "vessel_density": density,
        "vessel_length": length,
        "branching_points": branches,
        "fractal_dimension": fractal,
        "tortuosity": tort,
        "avg_thickness": thickness
    })

    print(f"Processed: {file}")

# -----------------------------
# SAVE CSV
# -----------------------------

df = pd.DataFrame(results)

df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Features saved to: {OUTPUT_CSV}")
