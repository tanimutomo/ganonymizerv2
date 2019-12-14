import os
import torch

from .edge_connect.src.edge_connect import SimpleEdgeConnect


class ImageInpainter():
    def __init__(self, opt):
        if opt.inpaint == 'EdgeConnect':
            self.model = self._set_edge_connect(opt)

    def inpaint(self, img, mask):
        return self.model.inpaint(img, mask)

    def _set_edge_connect(self, opt):
        inpaint_ckpt = 'modules/edge_connect/checkpoints'
        model = SimpleEdgeConnect(inpaint_ckpt, 
                                  opt.sigma,
                                  opt.device)
        return model

