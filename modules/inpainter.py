import os
import torch
from .edge_connect.src.edge_connect import SimpleEdgeConnect
from .utils import Debugger


class Inpainter():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.debugger = Debugger(config.inpaint_mode, save_dir=config.checkpoint)

        if self.config.inpaint == 'EdgeConnect':
            self.model = self._set_edge_connect()


    def inpaint(self, img, mask):
        output, out_edge, edge = self.model.inpaint(img, mask)
        return output, out_edge, edge


    def _set_edge_connect(self):
        model = SimpleEdgeConnect(
                self.config.inpaint_ckpt, self.config.sigma, self.device, self.debugger)
        return model

