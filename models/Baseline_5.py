import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_5(nn.Module):
    """
    backbone : resnet50 
    batch : clip
    padded players don't get fed to backbone instead they get get padded tensor with 0 as features 
    """
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(
        input_size=2048,      
        hidden_size=512,     
        num_layers=1,
        batch_first=True
        )
        
        self.classifier = nn.Sequential(nn.Linear(512, 8))
        
    def forward(self, x, mask):
        B, S, P, C, H, W = x.shape
        """
        B : batch size
        S : number of frames in the clip (9)
        P : number of players in the frame (12)
        C : number of channels (3)
        H : height (224)
        W : width (224)
        """
        x = x.view(B * S * P, C, H, W)
        mask = mask.view(B* S * P)

        valid_x = x[mask] # (B * S * real_players)  
        valid_features = self.backbone(valid_x)  # (B * S * real_players, 2048)
        valid_features = valid_features.flatten(1)
        
        features = torch.zeros(B * S * P, 2048, device=x.device, dtype=valid_features.dtype) # (B * S * P, 2048)
        features[mask] = valid_features # (B * S * P, 2048)
    
        features = features.view(B * S, P, -1)     # (B*S, P, 2048)  
        avg_pooling = features.mean(dim=1) # (B*S, 1, 2048)
        avg_pooling = avg_pooling.view(B, S, -1) # (B, S, 2048)

        lstm_out, _ = self.lstm(avg_pooling) # (B, S, 1024)
        lstm_out = lstm_out[:, -1, :] # (B, 1, 1024)
        lstm_out = lstm_out.squeeze(1) #(B, 1024)
        output = self.classifier(lstm_out) # (B, 8)
      
        return output
