from .encoderCNN import *
import sys
import torch.nn as nn
import os
import numpy as np
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GCNConv, BatchNorm
import torch.nn.functional as F
import torch


class optimNet(nn.Module):
    def __init__(self, cnn_encoder):
        super(optimNet, self).__init__()
        if cnn_encoder == 'efficientnet-b3':
            self.conv1 = GCNConv(1536, 512, improved=False, cached=False, bias=True)
            self.conv2 = GCNConv(512, 128, improved=False, cached=False, bias=True)
            self.mlp1 = nn.Sequential(
                nn.Linear(512 * 2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'efficientnet-b0':
            self.conv1 = GCNConv(1280, 512, improved=False, cached=False, bias=True)
            self.conv2 = GCNConv(512, 128, improved=False, cached=False, bias=True)
            self.mlp1 = nn.Sequential(
                nn.Linear(512 * 2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'densenet121':
            self.conv1 = GCNConv(1024, 512, improved=False, cached=False, bias=True)
            self.conv2 = GCNConv(512, 128, improved=False, cached=False, bias=True)
            self.mlp1 = nn.Sequential(
                nn.Linear(512 * 2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'resnet18' or cnn_encoder == 'resnet34':
            self.conv1 = GCNConv(512, 256, improved=False, cached=False, bias=True)
            self.conv2 = GCNConv(256, 128, improved=False, cached=False, bias=True)
            self.mlp1 = nn.Sequential(
                nn.Linear(256 * 2, 1),
                nn.ReLU()
            )
        elif cnn_encoder == 'resnet50' or cnn_encoder == 'resnet101':
            self.conv1 = GCNConv(2048, 1024, improved=False, cached=False, bias=True)
            self.conv2 = GCNConv(1024, 128, improved=False, cached=False, bias=True)
            self.mlp1 = nn.Sequential(
                nn.Linear(1024 * 2, 1),
                nn.ReLU()
            )
        else:
            print('Co to za sieć?')
            self.conv1 = GCNConv(1024, 512, improved=False, cached=False, bias=True)
            self.conv2 = GCNConv(512, 128, improved=False, cached=False, bias=True)

            self.mlp1 = nn.Sequential(
                nn.Linear(512*2, 1),
                nn.ReLU()
            )

    def similarity1(self, node_embedding, edge_index):
        edge_attr = []
        for i in range(len(edge_index[0])):
            x1 = self.mlp1(torch.cat((node_embedding[edge_index[0][i]], node_embedding[edge_index[1][i]]), 0))
            edge_attr.append(x1.reshape(1))
        edge_attr = torch.stack(edge_attr)
        return edge_attr

    def forward(self, node_attr, edge_attr, edge_index):
        node_embedding = node_attr
        out = self.conv1(node_embedding, edge_index, edge_attr.reshape(-1))
        out = F.relu(out)
        edge_attr = self.similarity1(out, edge_index)
        out = self.conv2(out, edge_index, edge_attr.reshape(-1))
        return out
