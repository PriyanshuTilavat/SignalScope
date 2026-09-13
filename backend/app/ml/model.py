import torch
import torch.nn as nn
from torchvision import models


class SignalScopeModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Use ImageNet-pretrained ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Replace original 1000-class classifier
        input_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            input_features,
            1
        )

    def forward(self, x):

        return self.backbone(x)


def create_model():

    return SignalScopeModel()
    