from transformers import CLIPModel
import torch.nn as nn


class BasicClipEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-large-patch14-336'):
        super(BasicClipEncoder, self).__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.image_encoder = self.clip.vision_model
        self.pooler = self.clip.visual_projection  # Linear projection

    def forward(self, images):
        outputs = self.image_encoder(pixel_values=images)
        pooled = outputs.pooler_output  # shape: [batch, hidden_dim]
        return self.pooler(pooled)