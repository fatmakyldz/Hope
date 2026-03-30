"""ResNet18 backbone for CIFAR (32x32 images)."""
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # CIFAR adaptation: replace first conv (7x7 stride 2 -> 3x3 stride 1)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        # Remove final classifier
        self.net = nn.Sequential(*list(resnet.children())[:-1])  # up to avgpool
        self.feature_dim = 512

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        out = self.net(x)           # (B, 512, 1, 1)
        return out.flatten(1)       # (B, 512)
