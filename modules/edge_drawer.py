import os
from .utils import Debugger


class EdgeDrawer():
    def __init__(self, config):
        self.config = config
        self.device = device
        self.debugger = Debugger(config.inpaint_mode, save_dir=config.checkpoint)

    def draw(self, img, mask):
        pass
