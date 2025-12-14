import pickle
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

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

        self.total_size = len(self.dataset)
        self.train_size = int(0.8 * self.total_size)
        self.test_size = self.total_size - self.train_size
        
        self.train_set, self.test_set = random_split(self.dataset, 
                                                     [self.train_size, self.test_size])
        
        self.train_loader = DataLoader(self.train_set, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=64, shuffle=False)

    # boucle d'entraînement
    def train(self):
        print("Beginning of the training of your network...")
        self.model.train()

        # dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True)
        for epoch in range(34):
            total_train_loss = 0
            train_correct = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):       
                # réinitialise la loss et les gradients du batch précédent
                self.optimizer.zero_grad()

                output = self.model(data)

                # calcule la loss par rapport à l'expert
                loss = self.criterion(output, target)

                total_train_loss += loss.item()

                train_pred = torch.argmax(output, dim=1)
                train_correct += (train_pred == target).sum().item()

                # met à jour les gradients de chaque paramètre du modèle
                # (poids du réseau)
                loss.backward()


                data_flip = torch.flip(data, [3])
                target_flip = target.clone()

                mask_east = (target == 2)
                mask_west = (target == 3)

                target_flip[mask_east] = 3
                target_flip[mask_west] = 2

                output_flip = self.model(data_flip)
                loss_flip = self.criterion(output_flip, target_flip)
                loss_flip.backward()

                total_train_loss += loss_flip.item()
                flip_pred = torch.argmax(output_flip, dim=1)
                train_correct += (flip_pred == target_flip).sum().item()
    
                
                # Met à jour les paramètres du modèle à partir 
                # du gradient calculé
                self.optimizer.step()
            
            avg_loss = total_train_loss / (2*len(self.train_loader))
            avg_accuracy = train_correct / (2*len(self.train_loader.dataset))

            # affiche la loss seulement une fois par epoch
            print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2%}")
                    
            # on sauvegarde à chaque epoch
            torch.save(self.model.state_dict(), "pacman_model.pth")

        # sauvegarde le modèle entraîné
        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Model saved !")

    def test(self):
        print("Beginning of the test of the model...")
        self.model.eval()

        test_loss = 0
        correct = 0 

        for data, target in self.test_loader:
            output = self.model(data)
            test_loss += self.criterion(output, target).item()
            pred = torch.argmax(output, dim=1)
            correct += (pred == target).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)

        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
    pipeline.test()
