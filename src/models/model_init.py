import torch
import torch.nn as nn
import torchvision.models as models
from .model import STN, ModifiedResnet, SiameseNetwork


def create_siamese_network(use_stn=True):
    #backbone = models.resnet50(pretrained=True)
    backbone = models.resnet50()
    state_dict = torch.load('/data/nas05/paul/BarlowTwins/checkpoint/resnet50.pth', map_location='cpu')
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)

    backbone.fc = nn.Sequential(
        nn.Linear(2048, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256, bias=True),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
    )
    modified_model = ModifiedResnet(backbone)
    stn_model = STN() if use_stn else None
    
    return SiameseNetwork(modified_model, stn_model)