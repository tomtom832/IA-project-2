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

        self.path = path
        self.dataset = PacmanDataset(self.path)
        
        # on initialise le modèle
        self.model = PacmanNetwork()

        # on charge la sauvegarde si elle existe
        save_path = "pacman_model.pth"
        if os.path.exists(save_path):
            print("Loading the model...")
            self.model.load_state_dict(torch.load(save_path, map_location='cpu'))
        else:
            print("Creation of a new model...")

        # on définit le critère et l'optimizer
        # le critère détermine comment on calcule si le réseau a raison ou tort
        # (si l'action qu'il prend est bonne ou pas par rapport à l'expert)
        self.criterion = nn.CrossEntropyLoss()
        # ça permet à l'optimizer de garder une trace des paramètres qui 
        # pourraient être mis à jour durant l'entraînement
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # boucle d'entraînement
    def train(self):
        print("Beginning of the training of your network...")

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True)
        for epoch in range(20):
            for batch_idx, (data, target) in enumerate(dataloader):
                # réinitialise la loss et les gradients du batch précédent
                self.optimizer.zero_grad()
                
                # calcule la loss par rapport à l'expert
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # met à jour les gradients de chaque paramètre du modèle
                # (poids du réseau)
                loss.backward()
                
                # Met à jour les paramètres du modèle à partir 
                # du gradient calculé
                self.optimizer.step()
                
            # affiche la loss seulement une fois par epoch
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                    
            # on sauvegarde à chaque epoch
            torch.save(self.model.state_dict(), "pacman_model.pth")

        # sauvegarde le modèle entraîné
        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Model saved !")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
