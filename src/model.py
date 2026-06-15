import torch
import torch.nn as nn
from torchvision import models

class DisasterCNN(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super().__init__()
        # Pretrained MobileNetV2 (ImageNet pe trained)
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Last layer ko apne 5 classes ke according replace karo
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)