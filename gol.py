import os

import cv2
import numpy as np


class GolManager:
    __xs = (0, 0, 0, 1, 1, 2, 2, 2)
    __ys = (0, 1, 2, 0, 2, 0, 1, 2)

    def __init__(self, gol_map, border_type=False, mut_prob=None):
        self.borders = border_type
        self.mut_prob = mut_prob
        self.gol_map = gol_map
        self.h, self.w = gol_map.shape

    def __get_rnd_map(self, r_sample):
        mut_map = np.random.choice(r_sample,
                                   size=self.h * self.w,
                                   p=[1 - self.mut_prob, self.mut_prob])

        return np.reshape(mut_map, (self.h, self.w))

    def __mutate(self):
        # revive some cells
        mutation_map = self.__get_rnd_map(r_sample=[0, 1])
        self.gol_map[mutation_map == 1] = 1

        # kill some cells
        mutation_map = self.__get_rnd_map(r_sample=[1, 0])
        self.gol_map[mutation_map == 0] = 0

    def __transparent_borders(self, pad_frame):
        last_col = self.gol_map[:, -1:]
        first_col = self.gol_map[:, [0]]

        first_row = self.gol_map[0, 0:]
        first_row = np.append(first_row, first_row[0])
        first_row = np.append(first_row[-2], first_row)

        last_row = self.gol_map[-1, 0:]
        last_row = np.append(last_row, last_row[0])
        last_row = np.append(last_row[-2], last_row)

        pad_frame[0, 0:] = last_row  # set first_row
        pad_frame[-1, 0:] = first_row  # set last_row
        pad_frame[1:self.h + 1, 0:1] = last_col  # set first_col
        pad_frame[1:self.h + 1, self.w + 1:self.w + 2] = first_col  # set last_col
        return pad_frame

    def update_board(self):
        if self.mut_prob > 0:
            self.__mutate()
        pad_frame = np.zeros((self.h + 2, self.w + 2))

        if self.borders == "transparent":
            pad_frame = self.__transparent_borders(pad_frame)

        pad_frame[1:1 + self.h, 1:1 + self.w] = self.gol_map

        # array with the sums of 8 nearby elements for each cell
        sum_frame_gen = (pad_frame[y:y + self.h, x:x + self.w]
                         for x, y in zip(self.__xs, self.__ys))
        mask = np.sum(sum_frame_gen, axis=0)
        self.gol_map[mask < 2] = 0
        self.gol_map[mask > 3] = 0
        self.gol_map[mask == 3] = 1


class GolMapCreator:

    @staticmethod
    def __normalize(bitmap):
        # invert if the background is light
        if np.mean(bitmap) > 100:
            bitmap = 255.0 - bitmap

        bitmap[bitmap < 50] = 0
        bitmap[bitmap > 0] = 1
        return bitmap

    def from_gif(self, path):
        cap = cv2.VideoCapture(path)
        for i in range(1):
            _, _ = cap.read()
        _, img = cap.read()
        cap.release()

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.__normalize(img)

    def from_image(self, path):
        return self.__normalize(cv2.imread(path, 0))

    @staticmethod
    def from_random(h, w):
        nums = np.random.choice([0, 1], size=h * w, p=[.6, .4])
        return np.reshape(nums, (h, w))

    def __init__(self, path=None, hw=tuple()):
        if path:
            if path.lower().endswith("gif"):
                self.gol_map = self.from_gif(path)
            else:
                self.gol_map = self.from_image(path)

        elif hw and len(hw) == 2:
            self.gol_map = self.from_random(*hw)
        else:
            raise Exception

        self.h, self.w = self.gol_map.shape


class GolMapVisualiser:
    def __init__(self, path, delay_ms, window_h):
        self.path = path
        self.window_h = window_h
        self.delay_ms = delay_ms
        self.__save_counter = 0
        self.running = True

    @staticmethod
    def __image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        h, w = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / h
            dim = int(w * r), height
        else:
            r = width / w
            dim = width, int(h * r)
        return cv2.resize(image, dim, interpolation=inter)

    @staticmethod
    def __np_to_cv(arr):
        gr_frame = arr.astype(np.uint8) * 255
        return cv2.cvtColor(gr_frame, cv2.COLOR_GRAY2RGB)

    def save(self, frame):
        filepath = os.path.join(self.path, f"frame_{self.__save_counter}.bmp")
        print(f"frame saved as: {filepath}")
        self.__save_counter += 1
        cv2.imwrite(filepath, frame)

    def show(self, gol_map):
        frame = self.__np_to_cv(gol_map)
        if self.window_h != 0:
            frame = self.__image_resize(frame,
                                        height=self.window_h,
                                        inter=cv2.INTER_NEAREST)
        cv2.imshow("GoL", frame)

        key = cv2.waitKey(self.delay_ms)

        # ESC pressed
        if key == 27:
            cv2.destroyAllWindows()
            print("ESC pressed")
            self.running = False

        # S pressed
        elif key == 115:
            self.save(frame)
