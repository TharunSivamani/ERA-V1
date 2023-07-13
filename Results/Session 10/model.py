import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv_layer1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_layer2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out += residual
        return out


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return x

class MakeLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,residual_block=None):
        super().__init__()
        self.residual_block = residual_block
        self.downsample_layer = DownsampleLayer(in_channels,out_channels,kernel_size,padding, stride)
        if residual_block:
            self.residual_layer = ResidualBlock(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size,
                                                stride=stride,padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layer(x)
        if self.residual_block:
            x = self.residual_layer(x)
        return x


class custom_resnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep_layer = ConvBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.layer1 = MakeLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                                residual_block=ResidualBlock)
        self.layer2 = MakeLayer(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
                                residual_block=None)
        self.layer3 = MakeLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                residual_block=ResidualBlock)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output