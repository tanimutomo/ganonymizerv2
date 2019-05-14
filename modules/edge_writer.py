import os
from .utils import Debugger


class EdgeWriter():
    def __init__(self, config):
        self.config = config
        self.device = device
        self.debugger = Debugger(config.inpaint_mode, save_dir=config.checkpoint)

    def write(self, img, mask):
        pass
