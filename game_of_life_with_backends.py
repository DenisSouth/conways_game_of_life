import time
from enum import Enum
from typing import Optional
import cv2
import numpy as np
from numba import njit, prange
import dask.array as da


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


@njit(parallel=True)
def _step_numba(grid, out, border_wrap):
    H, W = grid.shape
    for i in prange(H):
        for j in range(W):
            im1 = i - 1
            ip1 = i + 1
            jm1 = j - 1
            jp1 = j + 1
            if border_wrap:
                if im1 < 0: im1 += H
                if ip1 >= H: ip1 -= H
                if jm1 < 0: jm1 += W
                if jp1 >= W: jp1 -= W
                s = (grid[im1, jm1] + grid[im1, j] + grid[im1, jp1]
                     + grid[i, jm1] + grid[i, jp1]
                     + grid[ip1, jm1] + grid[ip1, j] + grid[ip1, jp1])
            else:
                s = 0
                if im1 >= 0:
                    if jm1 >= 0: s += grid[im1, jm1]
                    s += grid[im1, j]
                    if jp1 < W: s += grid[im1, jp1]
                if jm1 >= 0: s += grid[i, jm1]
                if jp1 < W: s += grid[i, jp1]
                if ip1 < H:
                    if jm1 >= 0: s += grid[ip1, jm1]
                    s += grid[ip1, j]
                    if jp1 < W: s += grid[ip1, jp1]
            c = grid[i, j]
            out[i, j] = 1 if (s == 3 and c == 0) or ((s == 2 or s == 3) and c == 1) else 0


class GameOfLife:
    __slots__ = ("grid", "mutation_rate", "border_mode", "backend", "_numba_scratch")

    _NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.uint8)
    _BORDER_MAP = {
        BorderMode.WRAP: cv2.BORDER_REFLECT,
        BorderMode.SOLID: cv2.BORDER_CONSTANT,
    }
    _DASK_BOUNDARY = {
        BorderMode.WRAP: "periodic",
        BorderMode.SOLID: 0,
    }

    def __init__(
            self,
            grid: np.ndarray,
            mutation_rate: float = 0.0,
            border_mode: BorderMode = BorderMode.WRAP,
            backend: str = "filter2d",
    ):
        self.grid = grid.astype(np.uint8, copy=False)
        self.mutation_rate = float(mutation_rate)
        self.border_mode = border_mode
        self.backend = backend
        if backend not in ("filter2d", "numba", "dask", "original"):
            raise ValueError(f"Unknown backend: {backend}")

        self._numba_scratch = np.empty_like(self.grid)

    def step(self) -> None:
        # mutation
        if self.mutation_rate > 0 and self.backend != "numba":
            # numba version does mutation inside JIT
            self._mutate()

        # neighbor count & update
        if self.backend == "filter2d":
            neighbours = cv2.filter2D(
                self.grid,
                ddepth=cv2.CV_8U,
                kernel=self._NEIGHBOR_KERNEL,
                borderType=self._BORDER_MAP[self.border_mode],
            ).astype(np.uint8)
            neighbours -= self.grid
            birth = (neighbours == 3) & (self.grid == 0)
            survive = ((neighbours == 2) | (neighbours == 3)) & (self.grid == 1)
            self.grid.fill(0)
            self.grid[birth | survive] = 1



        elif self.backend == "numba":
            if self.mutation_rate > 0:
                self._mutate()
            border_wrap = (self.border_mode is BorderMode.WRAP)
            _step_numba(self.grid, self._numba_scratch, border_wrap)
            self.grid, self._numba_scratch = self._numba_scratch, self.grid


        elif self.backend == "dask":
            depth = {0: 1, 1: 1}
            boundary = self._DASK_BOUNDARY[self.border_mode]
            dgrid = da.from_array(self.grid, chunks=(4096, 4096))
            def _block_step(block):
                block = block.astype(np.uint8)
                nb = cv2.filter2D(
                    block,
                    ddepth=cv2.CV_8U,
                    kernel=GameOfLife._NEIGHBOR_KERNEL,
                    borderType=GameOfLife._BORDER_MAP[self.border_mode],
                )
                nb = nb - block
                birth = (nb == 3) & (block == 0)
                survive = ((nb == 2) | (nb == 3)) & (block == 1)
                newb = np.zeros_like(block)
                newb[birth | survive] = 1
                return newb

            new = da.map_overlap(
                _block_step,
                dgrid,
                depth=depth,
                boundary=boundary,
                dtype=np.uint8,
            )

            result = new.compute()
            if self.mutation_rate > 0:
                # apply mutation on result
                flat = result.ravel()
                n = flat.size
                m = int(n * self.mutation_rate)
                idx = np.random.choice(n, 2 * m, replace=False)
                flat[idx[:m]] = 1
                flat[idx[m:]] = 0
            self.grid = result

        elif self.backend == "original":
            padded = np.pad(
                self.grid,
                pad_width=((1, 1), (1, 1)),
                mode=self.border_mode.value,
            )
            neighbours = sum(
                np.roll(np.roll(padded, dy, axis=0), dx, axis=1)[1:-1, 1:-1]
                for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                               (0, -1), (0, 1),
                               (1, -1), (1, 0), (1, 1)]
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
        seed: Optional[int],
        backend: str = "filter2d",
):
    if seed is None:
        seed = np.random.randint(0, 2 ** 32)
    np.random.seed(seed)
    print(f"SEED={seed} | Esc → exit | backend={backend}")

    cv2.namedWindow("Game of Life", cv2.WINDOW_NORMAL)
    if target_window_height > 0:
        scale = target_window_height / grid_height
        new_cols = int(grid_width * scale)
        cv2.resizeWindow("Game of Life", new_cols, target_window_height)

    initial_grid = create_map(size=(grid_height, grid_width), path=map_path)
    game = GameOfLife(initial_grid, mutation_rate, border_mode, backend=backend)

    while display(game.grid, frame_delay_ms, target_window_height):
        t0 = time.perf_counter()
        game.step()
        t1 = time.perf_counter()
        print(f"{BACKEND} Frame time: {(t1 - t0) * 1000:.2f} ms")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Grid dimensions
    GRID_HEIGHT = 10000
    GRID_WIDTH = 10000

    # Target display height (aspect ratio preserved)
    TARGET_WINDOW_HEIGHT = 1000

    # Border behavior: "wrap" (toroidal) or "solid" (dead edges)
    BORDER_MODE = BorderMode.SOLID

    # Probability of random mutation per cell per frame (0 = no mutation)
    MUTATION_RATE = 0.00000

    # Frame delay in milliseconds
    FRAME_DELAY_MS = 1

    # Initial map: random or predefined BMP
    MAP_PATH = LifeMaps.PUFFER_TRAIN_1.value

    # RNG seed (None = random seed on each run)
    SEED = None
    BACKEND = "numba"
    # BACKEND = "dask"
    # BACKEND = "filter2d"
    # BACKEND = "original"

    main(
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        target_window_height=TARGET_WINDOW_HEIGHT,

        border_mode=BORDER_MODE,
        mutation_rate=MUTATION_RATE,
        frame_delay_ms=FRAME_DELAY_MS,
        map_path=MAP_PATH,
        seed=SEED,
        backend=BACKEND,
    )
