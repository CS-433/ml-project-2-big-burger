import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data import random_split

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PaperCnn(nn.Module):
    def __init__(self):
        super(PaperCnn, self).__init__()
        self.layers = nn.Sequential(
            # Input: (N, 8, 64, 64)
            nn.Conv2d(8, 16, kernel_size=3, padding='same'),  # (N, 16, 64, 64)
            nn.BatchNorm2d(16),
            Swish(),
            nn.MaxPool2d(kernel_size=2),  # (N, 16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding='same'),  # (N, 32, 32, 32)
            nn.BatchNorm2d(32),
            Swish(),
            nn.MaxPool2d(kernel_size=2),  # (N, 32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding='same'),  # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            Swish(),

            nn.Conv2d(64, 64, kernel_size=3, padding='same'),  # (N, 64, 16, 16)
            nn.BatchNorm2d(64),
            Swish(),

            nn.Dropout(0.3),
            nn.Flatten(),  # Flatten for fully connected layer
            nn.Linear(64 * 16 * 16, 128),  # Adjust based on input size
            Swish(),
            nn.Linear(128, 1),  # Single output for regression
        )



    def forward(self, x):
        return self.layers(x)



def train_model_with_validation(model, images, Ds,  device, epochs=60, batch_size=32, learning_rate=0.01, momentum=0.9, 
        lr_drop_factor=0.1, 
        lr_drop_period=25, 
        validation_split=0.4
):
    """
    Train the model with SGD optimizer and validation.

    Parameters:
    - model: The CNN model.
    - images: A tensor of shape (N, 8, 1, 64, 64), where N is the number of samples.
    - Ds: A tensor of shape (N,), containing the true diffusion coefficients.
    - device: Training device (CPU or GPU).
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - learning_rate: Initial learning rate for SGD optimizer.
    - momentum: Momentum parameter for SGD.
    - lr_drop_factor: Factor by which the learning rate is reduced.
    - lr_drop_period: Period (in epochs) after which learning rate is reduced.
    - validation_split: Fraction of data to use for validation.

    Returns:
    - model: The trained model.
    - history: Dictionary with training and validation loss history.
    """
    # Split data into training and validation sets
    dataset = TensorDataset(images, Ds)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop_period, gamma=lr_drop_factor)

    history = {"train_loss": [], "val_loss": []}
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for batch_images, batch_Ds in train_loader:
            batch_images, batch_Ds = batch_images.to(device), batch_Ds.to(device)
            optimizer.zero_grad()
            predictions = model(batch_images)
            loss = criterion(predictions.squeeze(), batch_Ds)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_images.size(0)
        
        train_loss = running_train_loss / train_size
        history["train_loss"].append(train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_images, batch_Ds in val_loader:
                batch_images, batch_Ds = batch_images.to(device), batch_Ds.to(device)
                predictions = model(batch_images)
                loss = criterion(predictions.squeeze(), batch_Ds)
                running_val_loss += loss.item() * batch_images.size(0)
        
        val_loss = running_val_loss / val_size
        history["val_loss"].append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch progress
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, history

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