import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from timm.layers.attention import Attention

from src.model_retfound import build_retfound_model


def patch_attention_module(module):
    def forward_with_attention(x, attn_mask=None):
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        q = q * module.scale

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn)
        module.attn_weights = attn.detach().cpu()

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, module.attn_dim)
        x = module.norm(x)
        x = module.proj(x)
        x = module.proj_drop(x)
        return x

    module.forward = forward_with_attention


def run_attention():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "models/retfound_large_best.pth"
    LABELS_PATH = "outputs/predictions.csv"
    OUTPUT_DIR = "outputs/attention_maps"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    IMG_SIZE = 224

    print(f"Using device: {DEVICE}")
    print("Loading model...")

    model = build_retfound_model("pretrained/retfound_large.pth")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Invalid model checkpoint format")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "")
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    model = model.to(DEVICE)
    model.eval()

    attention_modules = []
    for module in model.modules():
        if isinstance(module, Attention):
            module.attn_weights = None
            patch_attention_module(module)
            attention_modules.append(module)

    print(f"✅ Patched {len(attention_modules)} transformer attention modules")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    df = pd.read_csv(LABELS_PATH)
    print(f"Loaded predictions.csv with {len(df)} rows")

    processed_count = 0

    for _, row in df.iterrows():
        img_path = str(row["image_path"])

        if not os.path.exists(img_path):
            print("❌ Missing image:", img_path)
            continue

        print("Processing:", img_path)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = model.forward_features(input_tensor)
            _ = model.head(features)

        attn_maps = []
        for module in attention_modules:
            if module.attn_weights is not None:
                attn_maps.append(module.attn_weights[0].numpy())

        if not attn_maps:
            print("⚠️ No attention maps captured for", img_path)
            continue

        attn_maps = [np.mean(att, axis=0) for att in attn_maps]
        joint_attn = np.eye(attn_maps[0].shape[0])

        for attn in attn_maps:
            attn = attn + np.eye(attn.shape[0])
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)
            joint_attn = attn @ joint_attn

        mask = joint_attn[0, 1:]
        grid_size = int(np.sqrt(mask.shape[0]))
        mask = mask.reshape(grid_size, grid_size)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask - mask.min()
        mask = mask / (mask.max() + 1e-8)

        original = cv2.imread(img_path)
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        retina_mask = gray > 10

        masked_heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        overlay_image = cv2.addWeighted(original, 0.6, masked_heatmap, 0.4, 0)

        overlay = np.zeros_like(original)
        overlay[retina_mask] = overlay_image[retina_mask]

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(OUTPUT_DIR, base_name + ".npy"), mask)

        cv2.imwrite(os.path.join(OUTPUT_DIR, base_name + "_overlay.jpg"), overlay)

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} images")

    print(f"✅ DONE — {processed_count} images processed!")


if __name__ == "__main__":
    run_attention()
