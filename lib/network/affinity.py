import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU


class affinityNet(torch.nn.Module):
    def __init__(self):
        super(affinityNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.mlp(inputs)
