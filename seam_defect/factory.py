from .models import FPNClassifier, ResNet18FPN
from torchvision import models
from torch import nn


def build_model(name, dropout, pretrained=True):
    if name == "resnet18_baseline":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, 11)
        return model
    if name == "resnet18_fpn":
        return ResNet18FPN(dropout=dropout, pretrained=pretrained)
    return FPNClassifier(name, num_classes=11, pretrained=pretrained, dropout=dropout)
