from enum import Enum
from typing import Optional

import cv2
import numpy as np


class BorderMode(Enum):
    WRAP = "wrap"
    SOLID = "constant"


class LifeMaps(Enum):
    RANDOM = ""
    STILL_OSC_SPACESHIPS = "map/still_osc_spaceships.bmp"
    OSCILLATORS = "map/oscillators.bmp"
    GOSPER_GUN_1 = "map/gosper_glider_gun_1.bmp"
    GOSPER_GUN_2 = "map/gosper_glider_gun_2.bmp"
    PUFFER_TRAIN_1 = "map/puffer_train_1.bmp"
    PUFFER_TRAIN_2 = "map/puffer_train_2.bmp"


class GameOfLife:
    __slots__ = ("grid", "mutation_rate", "border_mode")

    # (dy, dx) offsets for the eight neighbours
    _NEIGHBOR_SHIFTS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    def __init__(
            self,
            grid: np.ndarray,
            mutation_rate: float = 0.0,
            border_mode: BorderMode = BorderMode.WRAP,
    ):
        self.grid: np.ndarray = grid.astype(np.uint8, copy=False)
        self.mutation_rate: float = float(mutation_rate)
        self.border_mode: BorderMode = border_mode

    def step(self) -> None:
        if self.mutation_rate > 0.0:
            self._mutate()

        padded = np.pad(
            self.grid,
            pad_width=((1, 1), (1, 1)),
            mode=self.border_mode.value,
        )

        neighbours = sum(
            np.roll(np.roll(padded, dy, axis=0), dx, axis=1)[1:-1, 1:-1]
            for dy, dx in self._NEIGHBOR_SHIFTS
        )

        birth = (neighbours == 3) & (self.grid == 0)
        survive = ((neighbours == 2) | (neighbours == 3)) & (self.grid == 1)

        self.grid.fill(0)
        self.grid[birth | survive] = 1

    def _mutate(self) -> None:
        """Randomly kill and revive cells (in-place) according to mutation_rate."""
        n = self.grid.size
        m = int(n * self.mutation_rate)

        if m == 0:
            return

        flat = self.grid.ravel()
        idx = np.random.choice(n, 2 * m, replace=False)
        flat[idx[:m]] = 1  # revive
        flat[idx[m:]] = 0  # kill


def create_map(size: tuple[int, int], path: Optional[str] = None, offset_x: int = 20) -> np.ndarray:
    rows, cols = size
    if not path:
        return np.random.choice([0, 1], size=(rows, cols), p=[0.6, 0.4]).astype(np.uint8)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    h0, w0 = img.shape
    new_rows = max(rows, h0)
    new_cols = max(cols, w0)
    if (new_rows, new_cols) != (rows, cols):
        print(f"Expanded field to {new_rows}×{new_cols} to fit image")
    canvas = np.zeros((new_rows, new_cols), dtype=img.dtype)
    offset_y = (new_rows - h0) // 2
    canvas[offset_y:offset_y + h0, offset_x:offset_x + w0] = img
    img = canvas
    if np.mean(img) > 127:
        img = 255 - img
    _, binary = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def display(grid: np.ndarray, delay_ms: int, target_window_height: int) -> bool:
    frame = (grid * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Game of Life", frame)
    key = cv2.waitKey(delay_ms) & 0xFF
    if cv2.getWindowProperty("Game of Life", cv2.WND_PROP_VISIBLE) < 1:
        return False
    if key == 27:  # Esc
        return False
    return True


def main(
        grid_height: int,
        grid_width: int,
        target_window_height: int,
        border_mode: BorderMode,
        mutation_rate: float,
        frame_delay_ms: int,
        map_path: Optional[str],
        seed: Optional[int]):
    if seed is None:
        seed = np.random.randint(0, 2 ** 32)
    np.random.seed(seed)
    print(f"SEED={seed} | Esc → exit")

    cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
    if target_window_height > 0:
        scale = target_window_height / grid_height
        new_cols = int(grid_width * scale)
        cv2.resizeWindow("Game of Life", new_cols, target_window_height)

    initial_grid = create_map(size=(grid_height, grid_width), path=map_path)
    game = GameOfLife(initial_grid, mutation_rate, border_mode=border_mode,
                      )

    while display(game.grid, frame_delay_ms, target_window_height):
        game.step()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Grid dimensions
    GRID_HEIGHT = 3000
    GRID_WIDTH = 4000

    # Target display height (aspect ratio preserved)
    TARGET_WINDOW_HEIGHT = 1000

    # Border behavior: "wrap" (toroidal) or "solid" (dead edges)
    BORDER_MODE = BorderMode.WRAP

    # Probability of random mutation per cell per frame (0 = no mutation)
    MUTATION_RATE = 0.0001

    # Frame delay in milliseconds
    FRAME_DELAY_MS = 50

    # Initial map: random or predefined BMP
    MAP_PATH = LifeMaps.RANDOM.value

    # RNG seed (None = random seed on each run)
    SEED = None

    main(
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        target_window_height=TARGET_WINDOW_HEIGHT,

        border_mode=BORDER_MODE,
        mutation_rate=MUTATION_RATE,
        frame_delay_ms=FRAME_DELAY_MS,
        map_path=MAP_PATH,
        seed=SEED,
    )
