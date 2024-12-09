import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PaperCnnNoPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # Input: (N, 16, 64, 64) after temporal dimension adjustment
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # (N, 16, 32, 32)
            nn.BatchNorm2d(16),
            Swish(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (N, 32, 16, 16)
            nn.BatchNorm2d(32),
            Swish(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            Swish(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            Swish(),

            nn.Dropout(0.3),
            nn.Flatten(),  # Flatten for fully connected layer
            nn.Linear(64 * 16 * 16, 128),  # Adjust based on input size
            Swish(),
            nn.Linear(128, 1),  # Single output for regression
        )

    def forward(self, x):
        # Input: (N, 16, 1, 64, 64)
        x = x.squeeze(2)  # Remove the singleton channel dimension (N, 16, 64, 64)
        return self.layers(x)
