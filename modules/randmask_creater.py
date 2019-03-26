import cv2
import random
import numpy as np

from .utils import Debugger

class RandMaskCreater:
    def __init__(self, config):
        self.config = config
        self.debugger = Debugger(config.random_mode, save_dir=config.checkpoint)
        self.min_size = config.rmask_min
        self.max_size = config.rmask_max
    
    def sample(self, mask):
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        MAX_ITER = 100
        for idx in range(MAX_ITER):
            rmask = np.zeros_like(mask)
            shortd, longd, angle, cx, cy = self._sample_ellipse_coord(mask)
            self.debugger.matrix(rmask, 'rmask')
            rmask = cv2.ellipse(rmask, ((cx, cy), (longd, shortd), angle), 1, thickness=-1)
            self.debugger.matrix(mask, 'mask')
            self.debugger.matrix(rmask, 'rmask')
            if not np.any(mask + rmask > 1):
                break
            if idx == MAX_ITER - 1:
                return np.zeros_like(mask)

        return rmask.astype(np.uint8) * 255

    def _sample_ellipse_coord(self, mask):
        while True:
            short_diameter = random.randint(self.min_size, self.max_size)
            long_diameter = random.randint(self.min_size, self.max_size)
            if short_diameter < long_diameter:
                break
        angle = random.randint(0, 180)
        cx = random.randint(self.max_size, mask.shape[1] - self.max_size)
        cy = random.randint(self.max_size, mask.shape[0] - self.max_size)
        return short_diameter, long_diameter, angle, cx, cy


