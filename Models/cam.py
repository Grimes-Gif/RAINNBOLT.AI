import torch.nn as nn

class CAM(nn.Module):
    def __init__(self, pigeon_net, head="coords"):
        super().__init__()
        self.pigeon_net = pigeon_net
        self.head = head

    def forward(self, x):
        _coarse, _geo, coords = self.pigeon_net(x)
        if self.head == 'coords':
            return coords
        elif self.head == 'coarse':
            return _coarse
        else:
            return _geo
