import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from find_best_parameters import SimpleNN

if __name__ == "__main__":
    # Optimal hyperparameters found from find_best_parameters.py
    BEST_LEARNING_RATE = 0.001065741156220378
    BEST_BATCH_SIZE = 128
    BEST_EPOCHS = 40

    transform_list = transforms.Compose([
        transforms.RandomRotation(10),  # Small random rotations
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_list)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_list)

    # Data loaders with new batch size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BEST_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BEST_BATCH_SIZE, shuffle=True)

    # Model, loss function, and optimizer with the best hyperparameters
    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=BEST_LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    # Training loop with the best epochs
    for epoch in range(BEST_EPOCHS):
        current_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {current_loss/len(train_loader):.4f}')

    # Save the final trained model
    torch.save(model.state_dict(), "model/image_classifier.pt")