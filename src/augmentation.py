import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# -------------------------------------------------
# IMAGE
# -------------------------------------------------

IMAGE_PATH = "data/BRSET/fundus_photos/img00001.jpg"

image = Image.open(IMAGE_PATH).convert("RGB")

# -------------------------------------------------
# AUGMENTATIONS
# -------------------------------------------------

augmentations = {
    "Original": transforms.Compose([]),

    "Horizontal Flip":
        transforms.RandomHorizontalFlip(p=1),

    "Rotation":
        transforms.RandomRotation(20),

    "Brightness":
        transforms.ColorJitter(brightness=0.5),

    "Zoom":
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0)
        )
}

# -------------------------------------------------
# PLOT
# -------------------------------------------------

fig, axes = plt.subplots(
    1,
    5,
    figsize=(20, 5)
)

for ax, (title, aug) in zip(
    axes,
    augmentations.items()
):

    augmented = aug(image)

    ax.imshow(augmented)

    ax.set_title(title)

    ax.axis("off")

# -------------------------------------------------

plt.tight_layout()

plt.savefig(
    "outputs/augmentation_examples.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "✅ Saved augmentation visualization"
)
