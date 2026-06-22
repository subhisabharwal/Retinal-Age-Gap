import os
import torch
import torch.nn as nn
import importlib.util

# ===== FORCE LOAD RETFOUND MODEL =====
retfound_path = os.path.abspath("retfound/RETFound_MAE/models_vit.py")

spec = importlib.util.spec_from_file_location("models_vit", retfound_path)
models_vit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_vit)

vit_large_patch16 = models_vit.vit_large_patch16


def build_retfound_model(weight_path):
    model = vit_large_patch16(global_pool=True)

    # -------- LOAD WEIGHTS (robust) --------
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # remove "module." if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    print("✅ RETFound weights loaded")

    # -------- REGRESSION HEAD --------
    in_features = model.head.in_features

    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    return model


def unfreeze_partial(model):
    for name, param in model.named_parameters():
        if "blocks.20" in name or "blocks.21" in name or "blocks.22" in name or "blocks.23" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model
