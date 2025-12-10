import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """
    def __init__(self, H, W, hidden_dims, C, activation=nn.ReLU(), num_actions=5): 
        # num_actions = 5 car EST, OUEST, NORD, SUD, STOP
        super().__init__()

        # aplatir les dimensions du tenseur de data
        self.flat_input_size = C*H*W

        # créer la couche d'entrée
        self.mlp = [nn.Linear(self.flat_input_size, hidden_dims[0]), activation]

        # créer les couches cachées
        for i in range(len(hidden_dims) - 1):
            self.mlp += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), activation]

        # créer la couche de sortie
        head = [nn.Linear(hidden_dims[-1], num_actions)]

        # on met toutes les couches dans un réseau séquentiel
        self.mlp = nn.Sequential(*(self.mlp + head))

    def forward(self, x):
        """
        x de forme (B, C, H, W)
        """

        # Aplatir x
        # On garde la dim B (x.size(0)) et on écrase le reste (-1)
        # les dim de x deviennent donc (B, C*H*W)
        x = x.view(x.size(0), -1)

        # passage dans le mlp qui sort un vecteur de taille (B, num_actions)
        return self.mlp(x)
