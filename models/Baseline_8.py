import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Baseline_8(nn.Module):
    """
    backbone frozen and fined-tuned on person action
    batch : clip
    padded players aren't fed to backbone instead they get features of 0 and packed in tensors with existed players with same input order
    presence ratio for each team is fed to last layer as a feature (presence ratio for each team = number of real players / 12)
    note:(should've diveded on 6 not 12 in presence ratio but doesn't really matter since it is just a ratio) 
    """
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.backbone.parameters():
            param.requires_grad = False
                    
        self.lstm1 = nn.LSTM(
            input_size=2048,      
            hidden_size=512,     
            num_layers=1,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        self.classifier1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(.3)
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(130, 8)
        ) 
        
    def forward(self, x, mask):
        B, S, P, C, H, W = x.shape
        """
        B : batch size
        S : number of frames in clip (9)
        P : number of players in frame (12)
        C : number of channels (3)
        H : height (224)
        W : width (224)
        """
        x = x.view(B * S * P, C, H, W)

        first_frame_mask = mask[:, 0, :]
        team1_mask = first_frame_mask[:, :6]
        team2_mask = first_frame_mask[:, 6:]
        team1_count = team1_mask.sum(dim=1, keepdim=True).float()  
        team2_count = team2_mask.sum(dim=1, keepdim=True).float() 
        team1_presence_ratio = team1_count / P
        team2_presence_ratio = team2_count / P  
        
        mask = mask.view(B* S * P)

        valid_x = x[mask] # (B * S * real_players, C, H, W)  
        valid_features = self.backbone(valid_x)  # (B * S * real_players, 2048)
        valid_features = valid_features.flatten(1)
        
        features = torch.zeros(B * S * P, 2048, device=x.device, dtype=valid_features.dtype) # (B * S * P, 2048)
        features[mask] = valid_features # (B * S * P, 2048)
        
        features = features.view(B, S, P, -1)          # (B, S, P, 2048)
        features = features.permute(0, 2, 1, 3)        # (B, P, S, 2048) 
        features = features.reshape(B * P, S, -1)      # (B*P, S, 2048)
        
        lstm1_out, _ = self.lstm1(features)        # (B*P, S, 512)
        lstm1_out = lstm1_out.view(B, P, S, -1)    # (B, P, S, 512) 
        
        team1 = lstm1_out[:, :6, :, :].max(dim=1).values   # (B, S, 512)
        team2 = lstm1_out[:, 6:, :, :].max(dim=1).values   # (B, S, 512)
        two_teams = torch.cat([team1, team2], dim = 2) # (B, S, 1024)

        lstm2_out, _ = self.lstm2(two_teams) # (B, S, 1024)
        lstm2_out = lstm2_out[:, -1, :] # (B, 512)
        
        output = self.classifier1(lstm2_out)   # (B, 128)
        combined = torch.cat([output, team1_presence_ratio, team2_presence_ratio], dim=1) #(B, 129)
        output = self.classifier2(combined) # (B, 8)
        
        return output
