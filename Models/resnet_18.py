from torchvision.models import resnet18
import torch.nn as nn

class BackboneResNet(nn.Module):
    def __init__(self):
        super(BackboneResNet, self).__init__()
        model = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
