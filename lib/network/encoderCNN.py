from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet


class EncoderCNNdensenet(nn.Module):
    def __init__(self, embed_size=512):
        super(EncoderCNNdensenet, self).__init__()
        initial_cnn_temp = models.densenet121(pretrained=True)
        initial_cnn = []
        for child in initial_cnn_temp.children():
            initial_cnn.append(child)
        initial_cnn = initial_cnn[:-1]
        self.cnn = torch.nn.Sequential(*initial_cnn)
        for param in self.cnn.parameters():
            param.requires_grad = True

    def forward(self, images):
        out = self.cnn(images)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


class EncoderCNNresnet(nn.Module):
    def __init__(self, cnn_encoder):
        super(EncoderCNNresnet, self).__init__()
        if cnn_encoder == 'resnet18':
            initial_cnn_temp = models.resnet18(pretrained=True)
        elif cnn_encoder == 'resnet34':
            initial_cnn_temp = models.resnet34(pretrained=True)
        elif cnn_encoder == 'resnet50':
            initial_cnn_temp = models.resnet50(pretrained=True)
        elif cnn_encoder == 'resnet101':
            initial_cnn_temp = models.resnet101(pretrained=True)
        initial_cnn = []
        for child in initial_cnn_temp.children():
            initial_cnn.append(child)
        initial_cnn = initial_cnn[:-1]
        self.cnn = torch.nn.Sequential(*initial_cnn)
        for param in self.cnn.parameters():
            param.requires_grad = True

    def forward(self, images):
        out = self.cnn(images)
        out = torch.flatten(out, 1)
        return out

class EncoderCNNefficientnet(nn.Module):
    def __init__(self, cnn_encoder):
        super(EncoderCNNefficientnet, self).__init__()
        self.cnn = EfficientNet.from_pretrained(cnn_encoder)

        for param in self.cnn.parameters():
            param.requires_grad = True

    def forward(self, images):
        out = self.cnn.extract_features(images)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out
