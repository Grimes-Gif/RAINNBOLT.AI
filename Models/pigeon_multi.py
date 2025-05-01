import torch
import torch.nn as nn

class PigeonNet(nn.Module):
    def __init__(self, backbone, embedding_dim, num_coarse_geocells, num_geocells, dropout):
        """
        Args:
            backbone: Vision model (e.g., ResNet, ViT, CLIP encoder, etc.)
            embedding_dim: Output dimension of backbone features
            num_geocells: Number of geocells (classes)
        """
        super(PigeonNet, self).__init__()
        self.backbone = backbone
        
        self.neck = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout +.15),

            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU()
        )
        
        self.coarse_geocell_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_geocells)
        )

        # Classification head for geocell ID
        self.geocell_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_geocells)
        )

        # Regression head for (latitude, longitude) offset
        self.offset_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        
        x = self.backbone(x)
        
        x = self.neck(x)
        
        coarse_geocell_logits = self.coarse_geocell_head(x)
        geocell_logits = self.geocell_head(x)
        delta_coords = self.offset_head(x)

        return coarse_geocell_logits, geocell_logits, delta_coords
