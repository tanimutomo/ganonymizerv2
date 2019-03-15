import os
import torch
from .edge_connect.src.edge_connect import SimpleEdgeConnect
from .edge_connect.src.utils import Debugger


class Inpainter():
    def __init__(self, config, device, debugger):
        self.config = config
        self.device = device
        self.debugger = debugger

        if self.config['inpaint'] == 'EdgeConnect':
            self.model = self._set_edge_connect()


    def inpaint(self, img, mask):
        output, out_edge = self.model.inpaint(img, mask)
        # model.inpaint(self.config['checkpoints_path'], '/exp/inputs/places2_01.png')
        return output, out_edge


    def _set_edge_connect(self):
        model = SimpleEdgeConnect(
                self.config['inpaint_ckpt'], self.config['sigma'], self.device, self.debugger)
        return model


if __name__ == '__main__':
    from edge_connect.src.edge_connect import SimpleEdgeConnect
    from edge_connect.src.utils import Debugger
    config = {
            'checkpoints_path': 'edge_connect/checkpoints',
            'inpaint_network': 'EdgeConnect'
            }
    device = torch.device('cpu')
    debugger = Debugger(True, False)
    img = os.path.join(config['checkpoints_path'], 'exp/inputs/places2_01.png')
    inpainter = Inpainter(config, device, debugger)
    inpainter.inpaint(img)


