import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvPool2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPool2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class LocalizationNetwork(nn.Module):
    def __init__(self):
        super(LocalizationNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            ConvPool2D(3, 24),
            ConvPool2D(24, 32),
            ConvPool2D(32, 48),
            ConvPool2D(48, 64),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(12544, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        theta = self.fc_layers(x)
        theta = torch.tanh(theta) * torch.pi
        return theta

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = LocalizationNetwork()

    def get_rotation_matrix(self, theta):
        theta = theta.view(-1)
        batch_size = theta.size(0)
        rotation = torch.zeros(batch_size, 2, 3, device=theta.device)
        rotation[:, 0, 0] = torch.cos(theta)
        rotation[:, 0, 1] = -torch.sin(theta)
        rotation[:, 1, 0] = torch.sin(theta)
        rotation[:, 1, 1] = torch.cos(theta)
        return rotation

    def forward(self, x):
        theta = self.localization(x)
        rotation_matrix = self.get_rotation_matrix(theta)
        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        return x_transformed, theta

class ModifiedResnet(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResnet, self).__init__()
        self.features = base_model

    def forward(self, x):
        x = self.features(x)
        return F.normalize(x, p=2, dim=1)

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_model, stn):
        super(SiameseNetwork, self).__init__()
        self.stn_model = stn
        self.backbone_model = backbone_model

    def transform(self, x):
        if self.stn_model is not None:
            x, theta = self.stn_model(x)
        return x

    def forward(self, input1, input2):
        input1 = self.transform(input1)
        input2 = self.transform(input2)
        output1 = self.backbone_model(input1)
        output2 = self.backbone_model(input2)
        return output1, output2