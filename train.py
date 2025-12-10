import pickle
import os

import torch
import torch.nn as nn

from architecture import PacmanNetwork
from data import PacmanDataset


class Pipeline(nn.Module):
    def __init__(self, path):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        dataset = PacmanDataset(path)
        input, action = dataset[0]
        C, H, W = input.shape
        hidden_dims = [128, 64]

        self.path = path
        self.dataset = PacmanDataset(self.path)
        self.model = PacmanNetwork(H, W, hidden_dims, C)

        save_path = "pacman_model.pth"
        if os.path.exists(save_path):
            print("Chargement du modèle...")
            self.model.load_state_dict(torch.load(save_path))
        else:
            print("Création d'un nouveau modèle")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        print("Beginning of the training of your network...")

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=256, shuffle=True)
        for epoch in range(1000):
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Model saved !")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
