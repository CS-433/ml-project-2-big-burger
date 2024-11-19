import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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



# Training function
def train_model(model, images, Ds,device, epochs=20, batch_size=32, learning_rate=0.001):
    """
    Train the model to predict diffusion coefficient D.
    
    Parameters:
    - model: The CNN model.
    - images: A tensor of shape (N, 8, 1, 64, 64), where N is the number of samples.
    - Ds: A tensor of shape (N,), containing the true diffusion coefficients.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - learning_rate: Learning rate for the optimizer.
    - save_path: Path to save the trained model weights (optional).
    
    Returns:
    - model: The trained model.
    - loss_history: List of loss values per epoch.
    """
    # Set up DataLoader
    dataset = TensorDataset(images, Ds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_images, batch_Ds in dataloader:
            # Move to device (GPU or CPU)
            batch_images, batch_Ds = batch_images.to(device), batch_Ds.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_images)
            
            # Compute loss
            loss = criterion(predictions.squeeze(), batch_Ds)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_images.size(0)
        
        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    
    return model, loss_history

# Function to save model weights
def save_model_weights(model, path):
    """
    Save the model weights to a file.
    
    Parameters:
    - model: The CNN model.
    - path: Path to save the model weights.
    """
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

# Function to load model weights
def load_model_weights(model, path):
    """
    Load model weights from a file.
    
    Parameters:
    - model: The CNN model.
    - path: Path to the saved model weights.
    """
    model.load_state_dict(torch.load(path,weights_only=True))
    model.eval()  # Set the model to evaluation mode
    print(f"Model weights loaded from {path}")
    return model


def predict_diffusion_coefficients(model, images, device):
    """
    Predict the diffusion coefficient D for one or multiple images.

    Parameters:
    - model: The trained model.
    - images: A numpy array or tensor of shape (8, 64, 64) for a single image 
              or (N, 8, 64, 64) for multiple images.
    - device: The device (CPU or GPU) where the model is located.

    Returns:
    - predicted_Ds: A list of predicted diffusion coefficients.
    """
    # Ensure the images are a PyTorch tensor
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    
    # If the input is a single image, add the batch dimension
    if images.ndim == 3:  # Single image case
        images = images.unsqueeze(0)  # (8, 64, 64) -> (1, 8, 64, 64)
    
    # Add the channel dimension: (N, 8, 64, 64) -> (N, 8, 1, 64, 64)
    images = images.unsqueeze(2)
    
    # Move the images to the same device as the model
    images = images.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Get predictions
        predicted_Ds = model(images)
    
    # Return the predictions as a list
    return predicted_Ds.squeeze().tolist()

