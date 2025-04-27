import torch
import torch.nn as nn
import timm

class ViTGeoLocalization(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True, output_dim=2):
        """
        Args:
            model_name (str): Name of the ViT model from timm.
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            output_dim (int): Number of outputs. For geo, 2 (longitude, latitude).
        """
        super(ViTGeoLocalization, self).__init__()

        # Load pretrained ViT backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # <-- Remove original classifier
        )

        # Add your own regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)      # [batch_size, backbone_features]
        x = self.regressor(x)      # [batch_size, output_dim]
        return x