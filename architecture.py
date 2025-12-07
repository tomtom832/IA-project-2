import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """
    def __init__(self, H, W, C, hidden_dims, activation=nn.ReLU):
        super().__init__()

        self.flat_input_size = C*H*W

        self.mlp = [nn.Linear(H * W, hidden_dims[0]), activation]

        for i in range(len(hidden_dims) - 1):
            self.mlp += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), activation]

        head = [nn.Linear(hidden_dims[-1], H * W), nn.Sigmoid()]

        self.mlp = nn.Sequential(*(self.mlp + head))

        self.criterion = nn.BCELoss()

    def forward(self, x):
        # Your code here
        return # ...
