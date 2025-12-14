import torch
import torch.nn as nn
from data import PacmanDataset

# on récupère les dimensions du dataset
dataset = PacmanDataset("datasets/pacman_dataset.pkl")
input, action = dataset[0]
C, H, W = input.shape

# on définit les dim de hidden_dims
hidden_dims = [128, 64]


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """
    def __init__(self, C=C, H=H, W=W, n_kernels=32): 
        # num_actions = 5 car EST, OUEST, NORD, SUD, STOP
        
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=n_kernels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(n_kernels * (H//4) * (W//4), 50),
            nn.ReLU(),
            nn.Linear(50, C)
        )
        
    def forward(self, x):
        """
        x de forme (B, C, H, W)
        """

        return self.net(x)
