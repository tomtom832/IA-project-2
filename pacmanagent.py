from pacman_module.game import Agent

from data import state_to_tensor
from train import PacmanDataset
import torch


class PacmanAgent(Agent):
    def __init__(self, model):
        """
        Initialize the neural network Pacman agent.

        Arguments:
            model: The trained neural network model.
        """
        super().__init__()

        self.model = model.eval()
        
        self.INT_TO_ACTION = {
            0: 'North',
            1: 'South',
            2: 'East',
            3: 'West',
            4: 'Stop'
        }

    def get_action(self, state):
        """
        Return the action chosen by the neural network given the
        current state.

        Arguments:
            state: a GameState object
        """
        x = state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():   # on n'entraîne pas ⇒ pas de backprop
            logits = self.model(x)                 # prédiction brute (shape = (1,5))
            action_idx = torch.argmax(logits, dim=1).item()  # index de l'action (0–4)

        return self.INT_TO_ACTION[action_idx]
