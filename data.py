import pickle

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

    if state is None:
        raise ValueError("state must be a GameState object, got None")

    # Get grid-like objects from the state
    walls = state.getWalls()
    food = state.getFood()
    capsules = state.getCapsules()

    # Grid dimensions: Grid exposes width/height and is indexed as grid[x][y]
    width = walls.width
    height = walls.height

    # Create numpy arrays with shape (H, W) where row 0 corresponds to y=0
    walls_arr = np.zeros((height, width), dtype=np.float32)
    food_arr = np.zeros((height, width), dtype=np.float32)
    caps_arr = np.zeros((height, width), dtype=np.float32)
    pac_arr = np.zeros((height, width), dtype=np.float32)
    ghosts_arr = np.zeros((height, width), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            # place values at [y, x]
            walls_arr[y, x] = 1.0 if walls[x][y] else 0.0
            food_arr[y, x] = 1.0 if food[x][y] else 0.0

    for (x, y) in capsules:
        if 0 <= x < width and 0 <= y < height:
            caps_arr[y, x] = 1.0

    pac_pos = state.getPacmanPosition()
    if pac_pos is not None:
        px, py = map(int, pac_pos)
        if 0 <= px < width and 0 <= py < height:
            pac_arr[py, px] = 1.0

    ghost_positions = state.getGhostPositions()
    for (gx, gy) in ghost_positions:
        gx_i, gy_i = int(gx), int(gy)
        if 0 <= gx_i < width and 0 <= gy_i < height:
            ghosts_arr[gy_i, gx_i] += 1.0

    # Stack channels into (C, H, W)
    stacked = np.stack([walls_arr, food_arr, caps_arr, pac_arr, ghosts_arr], axis=0)

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
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        for s, a in data:
            x = state_to_tensor(s)
            self.inputs.append(x)
            self.actions.append(a)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]
