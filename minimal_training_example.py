"""
Minimal PyTorch Training Example

This script demonstrates a minimal training loop with PyTorch
to verify that the training environment is working properly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from torch.utils.data import Dataset, DataLoader

# Configure device - Auto-detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: No GPU detected. Training will be on CPU only.")

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples=10):
        self.data = torch.randn(num_samples, 10)
        self.labels = torch.randint(0, 5, (num_samples,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model():
    # Create model
    model = SimpleModel().to(device)
    print("Model created")
    
    # Create dataset and dataloader
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("Dataset created with 10 samples")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")
    
    print("Training complete!")
    
    # Save the model
    os.makedirs("./models/simple_model", exist_ok=True)
    torch.save(model.state_dict(), "./models/simple_model/model.pt")
    print("Model saved to ./models/simple_model/model.pt")

if __name__ == "__main__":
    print("Starting minimal PyTorch training example")
    try:
        train_model()
        print("Successfully completed training")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
