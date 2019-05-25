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
        # preprocess
        img, gray, mask, edge = self.model.preprocess(img, mask)
        # edge inpainting
        out_edge = self.model.edge_inpaint(gray, edge, mask)
        # color image inpainting
        output = self.model.image_inpaint(img, out_edge, mask)
        # postprocess
        edge = self.model.postprocess(edge)
        out_edge = self.model.postprocess(out_edge)
        output = self.model.postprocess(output)
        return output, out_edge, edge

    def color_inpaint(self, img, edge, mask):
        # preprocess
        img, edge, mask = self.model.cuda_tensor(img, edge, mask)
        # color image inpainting
        output = self.model.image_inpaint(img, edge, mask)
        # postprocess
        output = self.model.postprocess(output)
        return output

    def _set_edge_connect(self):
        inpaint_ckpt = os.path.join(self.config.mod_root, 'edge_connect/checkpoints')
        model = SimpleEdgeConnect(inpaint_ckpt, 
                                  self.config.sigma,
                                  self.device,
                                  self.debugger)
        return model

