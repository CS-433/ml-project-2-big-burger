import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3)  # Input channels: 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output for D
    
    def forward(self, x):
        # Input shape: (batch_size, timesteps=16, c=1, h=64, w=64)
        # Combine temporal dimension with channel dimension: (batch_size, timesteps * c, h, w)
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        
        # Apply convolutional layers
        x = self.relu(self.conv1(x))  # Output: (batch_size, 32, 32, 32)
        x = self.relu(self.conv2(x))  # Output: (batch_size, 64, 16, 16)
        x = self.relu(self.conv3(x))  # Output: (batch_size, 128, 8, 8)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Output: (batch_size, 128 * 8 * 8)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))  # Output: (batch_size, 128)
        x = self.relu(self.fc2(x))  # Output: (batch_size, 64)
        x = self.fc3(x)             # Output: (batch_size, 1)
        
        return x
