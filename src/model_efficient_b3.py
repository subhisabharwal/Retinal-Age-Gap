import timm
import torch.nn as nn


def build_model(pretrained=True):
    model = timm.create_model("efficientnet_b3", pretrained=pretrained)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

def unfreeze_partial(model):
    for name, param in model.named_parameters():
        if "blocks.5" in name or "blocks.6" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model
