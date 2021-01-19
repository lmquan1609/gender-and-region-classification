import torch
from torch import nn
from torchvision import models

class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, is_finetuned=False):
        super().__init__()
        self.model = models.densenet201(pretrained=pretrained)
        if is_finetuned:
            self.model.features.conv0.requires_grad = False
            self.model.features.norm0.requires_grad = False
            self.model.features.relu0.requires_grad = False
            self.model.features.pool0.requires_grad = False
            self.model.features.denseblock1.requires_grad = False
            self.model.features.transition1.requires_grad = False
            self.model.features.denseblock2.requires_grad = False
            self.model.features.transition2.requires_grad = False

        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output