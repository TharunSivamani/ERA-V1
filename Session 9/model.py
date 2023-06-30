import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        DROP = 0.01
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, groups=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Dropout(DROP),
            nn.ReLU(),

            nn.Conv2d(24, 24, 3, groups=24, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(24, 24, 1, groups=1, padding=0, bias=False)
        )

        self.trans1 = nn.Sequential(
            nn.Conv2d(24, 32, 3, groups=1, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(DROP),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, groups=32, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, groups=1, padding=0, bias=False),

            nn.Conv2d(32, 32, 3, groups=32, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, groups=1, padding=0, bias=False)
        )

        self.trans2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, groups=1, dilation=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(DROP),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, groups=64, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, groups=1, padding=0, bias=False),

            nn.Conv2d(64, 64, 3, groups=64, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, groups=1, padding=0, bias=False)
        )

        self.trans3 = nn.Sequential(
            nn.Conv2d(64, 96, 3, groups=1, dilation=4, bias=False),
            nn.BatchNorm2d(96),
            nn.Dropout(DROP),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, groups=96, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, groups=1, padding=0, bias=False),

            nn.Conv2d(96, 96, 3, groups=96, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.Conv2d(96, 96, 1, groups=1, padding=0, bias=False)
        )

        self.trans4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, groups=1, dilation=8, bias=False),
            nn.BatchNorm2d(96),
            nn.Dropout(DROP),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)
        x = x + self.conv2(x)
        x = self.trans2(x)
        x = x + self.conv3(x)
        x = self.trans3(x)
        x = x + self.conv4(x)
        x = self.trans4(x)
        x = self.output(x)
        return x