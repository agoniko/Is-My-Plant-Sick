# I modified the original implementations in order to be abgle to apply register forward hook and retrieve attention maps
# In those files I have also removed the registermodel
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from .torch_vit import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet50


class TorchVit(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        else:
            self.model = vit_b_16(weights=None)

        self.model.heads.head = nn.Linear(768, n_classes)

    def forward(self, x):
        return self.model(x)


class Resnet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = resnet50(weights="IMAGENET1K_V2")
        else:
            self.backbone = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.backbone(x).view(x.shape[0], -1)
        x = self.fc(x)
        return x
