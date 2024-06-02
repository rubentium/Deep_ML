import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from src.methods.deep_network import MyViT, Trainer

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Prepare DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define parameters for MyViT
chw = (1, 28, 28)  # MNIST images are 1 channel, 28x28 pixels
n_patches = 7  # Split each dimension into 7 patches, so each patch is 4x4 pixels
n_blocks = 6  # Number of transformer blocks
hidden_d = 64  # Hidden dimension of the model
n_heads = 8  # Number of attention heads
out_d = 10  # Number of output classes (digits 0-9)

# Instantiate the model
print("Instantiating the model")
model = MyViT(chw, n_patches, n_blocks, hidden_d, n_heads, out_d)
print("Model instantiated")

# Define the learning rate and number of epochs
lr = 1e-3
epochs = 10

# Instantiate the Trainer and train the model
trainer = Trainer(model, lr, epochs, batch_size)

print("Training the model")
trainer.train_all(train_loader)
print("Model trained")

# Evaluate the model on the test set
print("Evaluating the model")
pred_labels = trainer.predict_torch(test_loader)
test_labels = torch.cat([label for _, label in test_loader])
print("Model evaluated")

# Calculate accuracy
accuracy = (pred_labels == test_labels).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")