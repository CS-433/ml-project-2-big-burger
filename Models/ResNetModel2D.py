import torch.nn as nn

class ResNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),  # Increased filters to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Increased the number of residual blocks and filters
        self.res_block1 = ResidualBlock2D(64, 64)
        self.res_block2 = ResidualBlock2D(64, 128, stride=2)
        self.res_block3 = ResidualBlock2D(128, 256, stride=2)
        self.res_block4 = ResidualBlock2D(256, 512, stride=2)  # Added extra block

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)  # Increased the number of neurons

    def forward(self, x):
        # Input: (N, 16, 1, 64, 64)
        x = x.squeeze(2)  # Remove the singleton channel dimension (N, 16, 64, 64)
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)  # Additional block for more depth
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
