import numpy as np
import random
import torch

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent
from data import PacmanDataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path_to_saved_model = "pacman_model.pth"

# Feel free to add code here depending on your implementation
hidden_dims = [128, 64]
dataset = PacmanDataset("datasets/pacman_dataset.pkl")
input, action = dataset[0]
C, H, W = input.shape

model = PacmanNetwork(H, W, hidden_dims, C)
model.load_state_dict(torch.load(path_to_saved_model, map_location="cpu"))
model.eval()

pacman_agent = PacmanAgent(model)

score, elapsed_time, nodes = runGame(
    layout_name="test_layout",
    pacman=pacman_agent,
    ghosts=[SmartyGhost(1)],
    beliefstateagent=None,
    displayGraphics=True,
    expout=0.0,
    hiddenGhosts=False,
)

print(f"Score: {score}")
print(f"Computation time: {elapsed_time}")
