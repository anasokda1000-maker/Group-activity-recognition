import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_4(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
        input_size=2048,      
        hidden_size=1024,     
        num_layers=1, 
        batch_first=True,
        )
        
        self.classifier = nn.Sequential(           
            nn.Linear(1024, 8),
        )
        
    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
      
        features = self.backbone(x)
        features = features.view(B, S, -1)
      
        lstm_out, _ = self.lstm(features)
      
        predicted = self.classifier(lstm_out[:, -1, :])
      
        return predicted   # [B, num_classes]
