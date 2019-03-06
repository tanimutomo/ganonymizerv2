import os
import torch
from src.edge_connect import SimpleEdgeConnect
from src.utils import Debugger


class Inpainter():
    def __init__(self, config, device, debugger):
        self.config = config
        self.device = device
        self.debugger = debugger

        if self.config['inpaint_network'] == 'EdgeConnect':
            self.model = self._set_edge_connect()


    def inpaint(self, img):
        self.model.inpaint(img)
        # model.inpaint(self.config['checkpoints_path'], '/exp/inputs/places2_01.png')


    def _set_edge_connect(self):
        sigma = 2
        model = SimpleEdgeConnect(
                self.config['checkpoints_path'], sigma, self.device, self.debugger)
        return model


if __name__ == '__main__':
    config = {
            'checkpoints_path': 'checkpoints',
            'inpaint_network': 'EdgeConnect'
            }
    device = torch.device('cpu')
    debugger = Debugger(True, False)
    img = os.path.join(config['checkpoints_path'], 'exp/inputs/places2_01.png')
    inpainter = Inpainter(config, device, debugger)
    inpainter.inpaint(img)


