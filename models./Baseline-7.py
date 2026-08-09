import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_7(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=2048,      
            hidden_size=1024,     
            num_layers=1,
            batch_first=True
        )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(129, 8)
        )
        
    def forward(self, x, mask):
        B, S, P, C, H, W = x.shape
        x = x.view(B * S * P, C, H, W)

        counts = mask.float().mean(dim=1).sum(dim=1, keepdim=True)  # (B, 1)
        presence_ratio = counts / P
        
        mask = mask.view(B* S * P)
        
        valid_x = x[mask] # (B * S * real_players)  
        valid_features = self.backbone(valid_x)  # (B * S * real_players, 2048)
        valid_features = valid_features.flatten(1)
        features = torch.zeros(B * S * P, 2048, device=x.device, dtype=valid_features.dtype) # (B * S * P, 2048)
        features[mask] = valid_features # (B * S * P, 2048)
         
        features = features.view(B, S, P, -1)          # (B, S, P, 2048)
        features = features.permute(0, 2, 1, 3)         # (B, P, S, 2048) —
        features = features.reshape(B * P, S, -1)        # (B*P, S, 2048)
        
        lstm_out, _ = self.lstm(features)                # (B*P, S, 1024)
        lstm_out = lstm_out[:, -1, :]                     # (B*P, 1024) 
        
        lstm_out = lstm_out.view(B, P, -1)                # (B, P, 1024)
        max_pooling = lstm_out.max(dim=1).values              # (B, 1024) 
        
        output = self.classifier1(max_pooling) # (B, 128)
        combined = torch.cat([output, presence_ratio], dim=1)  # (B, 129)
        output = self.classifier2(combined) # (B, 8)

        return output
