import os
import torch
from .edge_connect.src.edge_connect import SimpleEdgeConnect
from .utils import Debugger


class ImageInpainter():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.debugger = Debugger(config.inpaint_mode, save_dir=config.checkpoint)

        if self.config.inpaint == 'EdgeConnect':
            self.model = self._set_edge_connect()

    def inpaint(self, img, mask):
        return self.model.inpaint(img, mask)

    def _set_edge_connect(self):
        inpaint_ckpt = os.path.join(self.config.mod_root,
                                    'edge_connect/checkpoints')
        model = SimpleEdgeConnect(inpaint_ckpt, 
                                  self.config.sigma,
                                  self.device,
                                  self.debugger)
        return model

