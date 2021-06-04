import torch
from torch import nn


class affinity_appearanceNet(torch.nn.Module):
    def __init__(self, cnn_encoder):
        super(affinity_appearanceNet, self).__init__()
        if cnn_encoder == 'efficientnet-b3':
            self.mlp = nn.Sequential(
                nn.Linear(1536*2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'efficientnet-b0':
            self.mlp = nn.Sequential(
                nn.Linear(1280*2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'densenet121':
            self.mlp = nn.Sequential(
                nn.Linear(1024*2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'resnet18' or cnn_encoder == 'resnet34':
            self.mlp = nn.Sequential(
                nn.Linear(512*2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'resnet50' or cnn_encoder == 'resnet101':
            self.mlp = nn.Sequential(
                nn.Linear(2048*2, 1),
                nn.ReLU()
            )
        else:
            print('Co to za sieć?')
            self.mlp = nn.Sequential(
                nn.Linear(1024*2, 1),
                nn.ReLU()
            )

    def forward(self, inputs):
        return self.mlp(inputs)
