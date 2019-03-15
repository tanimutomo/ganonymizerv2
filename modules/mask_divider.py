import cv2
import numpy as np

from .utils import expand_mask, detect_object, Debugger


class MaskDivider:
    def __init__(self, config, inpainter):
        self.config = config
        self.debugger = Debugger(config['divide_mode'], save_dir=config['checkpoint'])
        self.inpainter = inpainter
    

    def divide(self, img, mask):
        pass
