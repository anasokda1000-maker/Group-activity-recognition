import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_4(nn.Module):
    """
    backbone : resnet50 
    batch : clip
    """
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
        """
        B : batch size
        S : number of frames in the clip (9)
        C : number of channels (3)
        H : height (224)
        W : width (224)
        """
        x = x.view(B * S, C, H, W)
      
        features = self.backbone(x) # (B*S, 2048)
        features = features.view(B, S, -1)
      
        lstm_out, _ = self.lstm(features) # (B*S, 1024)
                                          # fed to lstm for the frames sequence dynamics 
      
        predicted = self.classifier(lstm_out[:, -1, :])
      
        return predicted   # [B, 8]
