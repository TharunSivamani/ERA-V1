import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dropout):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            Make_Custom_Layer(3, 64, pool=False, size=0, dropout=dropout),
            Make_Custom_Layer(64, 128, pool=True, size=2, dropout=dropout),
            Make_Custom_Layer(128, 256, pool=True, size=0, dropout=dropout),
            Make_Custom_Layer(256, 512, pool=True, size=2, dropout=dropout),
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x)
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, bias=False, stride=1, padding=1, pool=False):
        super(ConvLayer, self).__init__()

        block = list()
        block.append(nn.Conv2d(in_channels, out_channels, 3, bias=bias, stride=stride, padding=padding, padding_mode='replicate'))
        if pool:
            block.append(nn.MaxPool2d(2, 2))
        block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU())
        block.append(nn.Dropout(dropout))

        self.blocks = nn.Sequential(*block)

    def forward(self, x):
        return self.blocks(x)


class Make_Custom_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, pool=True, size=2):
        super(Make_Custom_Layer, self).__init__()

        self.dropout = dropout
        self.pool_block = ConvLayer(in_channels, out_channels, pool=pool, dropout=dropout)
        self.res_block = None
        if size > 0:
            layer = list()
            for i in range(0, size):
                layer.append(
                    ConvLayer(out_channels, out_channels, pool=False, dropout=dropout)
                )
            self.res_block = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pool_block(x)

        if self.res_block is not None:
            Y = x
            x = self.res_block(x)
            x = x + Y
        return x