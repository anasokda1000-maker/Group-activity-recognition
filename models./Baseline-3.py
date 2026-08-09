import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_B3_tuned(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        B, P, C, H, W = x.shape

        x = x.view(B * P, C, H, W)

        features = self.backbone(x)

        features = features.view(B, P, -1)

        pooled_features, _ = torch.max(features, dim=1)

        output = self.classifier(pooled_features)
        
        return output # tensor.size([B, 8])
