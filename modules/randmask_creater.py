import cv2
import random
import numpy as np

from .utils import Debugger

class RandMaskCreater:
    def __init__(self, config):
        self.config = config
        self.debugger = Debugger(config.random_mode, save_dir=config.checkpoint)
    
    def sample(self, mask):
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        MAX_ITER = 100
        for idx in range(MAX_ITER):
            rmask = np.zeros_like(mask)
            self.debugger.matrix(rmask, 'rmask')
            # ellipse mask
            if self.config.rmask_shape == 'ellipse':
                shortd, longd, angle, cx, cy = self._sample_ellipse_coord(mask)
                rmask = cv2.ellipse(rmask, ((cx, cy), (longd, shortd), angle), 1, thickness=-1)
            # rectangle mask
            elif self.config.rmask_shape == 'rectangle':
                tl, br = self._sample_rectangle_coord(self, mask)
                rmask = cv2.rectangle(rmask, tl, br, 1, -1)
            self.debugger.matrix(mask, 'mask')
            self.debugger.matrix(rmask, 'rmask')
            if not np.any(mask + rmask > 1):
                break
            if idx == MAX_ITER - 1:
                return np.zeros_like(mask)

        return rmask.astype(np.uint8) * 255

    def _sample_ellipse_coord(self, mask):
        min_diameter = int(self.config.min_size / 2)
        max_diameter = int(self.config.max_size / 2)
        while True:
            short_diameter = random.randint(min_diameter, max_diameter)
            long_diameter = random.randint(min_diameter, max_diameter)
            if short_diameter < long_diameter:
                break
        angle = random.randint(0, 180)
        cx = random.randint(max_diameter, mask.shape[1] - max_diameter)
        cy = random.randint(max_diameter, mask.shape[0] - max_diameter)
        return short_diameter, long_diameter, angle, cx, cy

    def _sample_rectangle_coord(self, mask):
        side_len = random.randint(self.config.min_size, self.config.max_size)
        cx = random.randint(self.config.max_size, mask.shape[1] - self.config.max_size)
        cy = random.randint(self.config.max_size, mask.shape[0] - self.config.max_size)
        half_len = int(side_len / 2)
        tl = (cx - half_len, cy - half_len)
        br = (cx + half_len, cy + half_len)
        return tl, br



