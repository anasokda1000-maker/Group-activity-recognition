import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_6(nn.Module):
    """
    backbone frozen and fined-tuned on person action
    batch : clip
    padded players aren't fed to backbone instead they get features of 0 and packed in tensors with existed players with same input order
    presence ratio is fed to last layer as a feature (presence ratio = number of real players / 12)
    """
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=2048,      
            hidden_size=512,     
            num_layers=1,
            batch_first=True
        )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier2 = nn.Sequential(
            nn.Linear(129,8)
        )
        
    def forward(self, x, mask):
        B, S, P, C, H, W = x.shape
        x = x.view(B * S * P, C, H, W)
        
        first_frame_mask = mask[:, 0, :]
        mask = mask.view(B* S * P)

        valid_x = x[mask] # (B * S * real_players)  
        valid_features = self.backbone(valid_x)  # (B * S * real_players, 2048)
        valid_features = valid_features.flatten(1)
        
        features = torch.zeros(B * S * P, 2048, device=x.device, dtype=valid_features.dtype) # (B * S * P, 2048)
        features[mask] = valid_features # (B * S * P, 2048)
    
        features = features.view(B * S, P, -1)     # (B*S, P, 2048)  
        pooled = features.max(dim=1).values  # (B*S, 2048)
    
        # real players ratio per clip 
        counts = first_frame_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        presence_ratio = counts / P  
    
        pooled = pooled.view(B, S, -1)  # (B, S, 2048)
    
        lstm_out, _ = self.lstm(pooled)
        lstm_out = lstm_out[:, -1, :] # (B, 512)
        
        output = self.classifier1(lstm_out)
        combined = torch.cat([output, presence_ratio], dim=1)  # (B, 129)
        output = self.classifier2(combined)
                                
        return output
