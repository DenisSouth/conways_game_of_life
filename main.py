import os

import numpy as np
from dotenv import load_dotenv

from gol import GolManager, GolMapVisualiser, GolMapCreator

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

MAP_PATH = os.environ.get("MAP_PATH", None)
W = int(os.environ.get("W", 600))
H = int(os.environ.get("H", 650))

BORDER_TYPE = os.environ.get("BORDER_TYPE", "transparent")
MUTATION_PROB = float(os.environ.get("MUTATION_PROB", 0))
SEED = int(os.environ.get("SEED", -1))
DELAY = int(os.environ.get("DELAY", 1))
WINDOW_H = int(os.environ.get("WINDOW_H", 600))
SAVE_PATH = os.environ.get("SAVE_PATH", "")

if __name__ == "__main__":
    if SEED != -1:
        np.random.seed(SEED)
    else:
        SEED = np.random.randint(0, 2 ** 32)
        np.random.seed(SEED)
        print(f"SEED set to {SEED}")
    print(f'stop > "ESC", save frame > "S"')

    gol_first_frame = GolMapCreator(path=MAP_PATH, hw=(H, W)).gol_map
    gol_manager = GolManager(gol_map=gol_first_frame,
                             border_type=BORDER_TYPE,
                             mut_prob=MUTATION_PROB)

    visualiser = GolMapVisualiser(path=SAVE_PATH, delay_ms=DELAY, window_h=WINDOW_H)
    visualiser.show(gol_manager.gol_map)

    while visualiser.running:
        gol_manager.update_board()
        visualiser.show(gol_manager.gol_map)
