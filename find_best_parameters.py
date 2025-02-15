import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import optuna # A hyperparameter optimization library for automating the search for the best hyperparameters.

# Defining a model which inherits from the torch.nn.Module, which allows us to make use of various PyTorch functions.
# Fully connected neural network.
# Skips the convolutions stage that would typically feed into a FCNN.
# This is because the dataset is rather simple, so going straight to prediction would be faster.
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

# Cross-validation setup
def get_data_loaders(trainset, batch_size, validation_split=0.2):
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(trainset, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader

# Objective function for Optuna
def objective(trial):
    # Hyperparameter suggestions
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

    # Create model and optimizer
    model = FCNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Get data loaders with cross-validation
    train_loader, val_loader = get_data_loaders(trainset, batch_size)

    # Training loop
    for epoch in range(40):  # Diminishing returns after 40 epochs
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward() # back propagation
            optimizer.step()

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # Loss function
    loss_function = nn.CrossEntropyLoss()

    transform_list = transforms.Compose([
        transforms.RandomRotation(10),  # Small random rotations
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Load the MNIST dataset
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_list)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_list)

    # Run the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Print the best hyperparameters
    print("Best Hyperparameters:", study.best_params)
