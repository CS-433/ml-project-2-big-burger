import torch.nn as nn

class DiffusionPredictorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Downsample by factor of 2
        self.relu = nn.ReLU()
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(input_size=64 * 8 * 8, hidden_size=128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Single output for D
    
    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        
        # Apply convolutional layers on each frame
        x = x.view(batch_size * timesteps, c, h, w)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # Final size: (batch_size * timesteps, 64, 8, 8)
        
        # Flatten spatial dimensions
        x = x.view(batch_size, timesteps, -1)
        
        # LSTM for temporal processing
        x, _ = self.lstm(x)  # Output size: (batch_size, timesteps, 128)
        
        # Take the last timestep output
        x = x[:, -1, :]  # Size: (batch_size, 128)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Size: (batch_size, 1)
        
        return x
    