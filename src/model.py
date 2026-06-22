import torch.nn as nn
import timm


def build_model(model_name="efficientnet_b0", pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)

    in_features = model.classifier.in_features

    # Replace classifier for regression
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )

    return model


# OPTIONAL: Freeze backbone
def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# OPTIONAL: Unfreeze everything
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

    return model
