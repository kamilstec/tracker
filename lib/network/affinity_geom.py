import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU


class affinity_geomNet(torch.nn.Module):
    def __init__(self):
        super(affinity_geomNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 1),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.mlp(inputs)
