# model.py

import torch.nn as nn

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
