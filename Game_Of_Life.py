import numpy as np
from typing import Literal


class GameOfLife:
    __slots__ = ("grid", "border_function", "mutations")
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def __init__(self, height: int, width: int, border_type: Literal['solid', 'wrap'], mutations=0.0):
        self.mutations = mutations
        self.grid = np.random.randint(2, size=(height, width), dtype=np.uint8)
        self.border_function = self.set_solid_border if border_type == 'solid' else lambda: ...

    def mutate(self):
        if self.mutations > 0:
            mut = self.mutations / 2
            grid_size = self.grid.size
            indices = np.random.choice(np.arange(grid_size), int(grid_size * mut), replace=False)
            self.grid[np.unravel_index(indices, self.grid.shape)] = 1
            indices = np.random.choice(np.arange(grid_size), int(grid_size * mut), replace=False)
            self.grid[np.unravel_index(indices, self.grid.shape)] = 0

    def set_solid_border(self):
        self.grid[0, :], self.grid[-1, :], self.grid[:, 0], self.grid[:, -1] = 0, 0, 0, 0

    def iterate(self):
        self.border_function()
        neighbors = sum(np.roll(self.grid, shift=s, axis=(0, 1)) for s in self.shifts)
        self.grid[(self.grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0
        self.grid[(self.grid == 0) & (neighbors == 3)] = 1


def show():
    from matplotlib import pyplot as plt
    figure = plt.figure()
    image = plt.imshow(game_of_life.grid, cmap='gray', vmin=0, vmax=1)

    while True:
        game_of_life.iterate()
        image.set_data(game_of_life.grid)
        plt.pause(0.0001)
        figure.canvas.draw()


def to_gif(path='game_of_life.gif', duration=1000):
    from PIL import Image
    frames = []
    for i in range(duration):
        frame = Image.fromarray(game_of_life.grid * 255)
        frames.append(frame)
        game_of_life.iterate()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration // 20, loop=0)


# game_of_life = GameOfLife(height=600, width=800, border_type="solid", mutations=0.0)
game_of_life = GameOfLife(height=600, width=800, border_type="wrap", mutations=0.01)

# to_gif()
show()

# from scipy.signal import convolve2d
# boundary = 'fill'
# boundary = 'wrap'
# neighbors = convolve2d(self.grid, np.ones((3, 3)), mode='same', boundary=boundary, fillvalue = 0) - self.grid
