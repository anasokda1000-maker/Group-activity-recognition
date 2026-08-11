import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class person_action(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 9)
        )

    def forward(self, x):
            B, C, H, W = x.shape
        
            features = self.backbone(x)  # [B, 9]
    
            return features
