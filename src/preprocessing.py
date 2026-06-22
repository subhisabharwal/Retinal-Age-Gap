import cv2
import numpy as np
from torchvision import transforms


# =========================================
# 1. CROP BLACK BACKGROUND
# =========================================
def crop_black_background(img, tol=7):
    """
    Removes black borders from retinal images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    if mask.any():
        img = img[np.ix_(mask.any(1), mask.any(0))]

    return img


# =========================================
# 2. CLAHE (CONTRAST ENHANCEMENT)
# =========================================
def apply_clahe(img):
    """
    Apply CLAHE to enhance contrast (mild version).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img


# =========================================
# 3. RESIZE + PAD (NO DISTORTION)
# =========================================
def resize_and_pad(img, size=224):
    """
    Resize image while maintaining aspect ratio
    and pad to square.
    """
    h, w, _ = img.shape

    scale = size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    img_resized = cv2.resize(img, (new_w, new_h))

    padded_img = np.zeros((size, size, 3), dtype=np.uint8)

    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2

    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized

    return padded_img


# =========================================
# 4. FULL PREPROCESS FUNCTION
# =========================================
def preprocess_image(img, size=224):
    """
    Complete preprocessing pipeline:
    Crop → CLAHE → Resize + Pad
    """
    img = crop_black_background(img)
    img = apply_clahe(img)
    img = resize_and_pad(img, size=size)

    return img


# =========================================
# 5. TRAIN TRANSFORMS (WITH AUGMENTATION)
# =========================================
def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =========================================
# 6. VALIDATION / TEST TRANSFORMS
# =========================================
def get_test_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
