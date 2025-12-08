import pickle
import numpy as np

import torch
from torch.utils.data import Dataset


def state_to_tensor(state):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    # Basic, deterministic feature extraction from GameState.
    # Returns a FloatTensor of shape (C, H, W) where channels are:
    # 0: walls, 1: food, 2: capsules, 3: pacman, 4: ghosts
    # If you want additional scalar features (score, remaining food, ...),
    # consider returning them separately or appending as extra channels.

    # sécurité 
    if state is None:
        raise ValueError("state must be a GameState object, got None")

    # murs
    walls = state.getWalls()
    # nourriture/petits points
    food = state.getFood()
    # capsules/gros points
    capsules = state.getCapsules()

    # Dim grille 
    width = walls.width
    height = walls.height

    # arrays de taille (H, W) (row 0 corresponds to y=0)
    walls_arr = np.zeros((height, width), dtype=np.float32)
    food_arr = np.zeros((height, width), dtype=np.float32)
    caps_arr = np.zeros((height, width), dtype=np.float32)
    pac_arr = np.zeros((height, width), dtype=np.float32)
    ghosts_arr = np.zeros((height, width), dtype=np.float32)


    # positionnement des murs et nourriture
    for x in range(width):
        for y in range(height):
            walls_arr[y, x] = 1.0 if walls[x][y] else 0.0
            food_arr[y, x] = 1.0 if food[x][y] else 0.0


    # positionnement des capsules comprises dans la grille
    for (x, y) in capsules:
        if 0 <= x < width and 0 <= y < height:
            caps_arr[y, x] = 1.0

    # positionnement de pacman
    pac_pos = state.getPacmanPosition()
    if pac_pos is not None:
        px, py = map(int, pac_pos)
        if 0 <= px < width and 0 <= py < height:
            pac_arr[py, px] = 1.0


    # positionnement des fantomes
    ghost_positions = state.getGhostPositions()
    for (gx, gy) in ghost_positions:
        gx_i, gy_i = int(gx), int(gy)
        if 0 <= gx_i < width and 0 <= gy_i < height: 
            ghosts_arr[gy_i, gx_i] += 1.0 # au cas où plusieurs fantomes au mm endroit

    # Stack des channels en (C, H, W) comme ds notebook
    stacked = np.stack([walls_arr, food_arr, caps_arr, pac_arr, ghosts_arr], axis=0)

    # conversion en tensor float
    tensor = torch.from_numpy(stacked).float()
    return tensor


class PacmanDataset(Dataset):
    def __init__(self, path):
        """
        Load and transform the pickled dataset into a format suitable
        for training your architecture.

        Arguments:
            path: The file path to the pickled dataset.
        """
        # Load the pickled data
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        for s, a in data:
            x = state_to_tensor(s)
            self.inputs.append(x)
            self.actions.append(a)

    # longueur du dataset
    def __len__(self):
        return len(self.inputs)

    # méthode pour obtenir un échantillon
    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]
