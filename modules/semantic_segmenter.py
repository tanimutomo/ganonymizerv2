import os
import torch
from .deeplabv3.model.deeplabv3 import DeepLabV3
from .utils import tensor_img_to_numpy, Debugger

class SemanticSegmenter():
    def __init__(self, config, device):
        self.network = config.semseg
        self.resnet = config.resnet
        self.device = device
        self.debugger = Debugger(config.semseg_mode, save_dir=config.checkpoint)

        if self.network == 'DeepLabV3':
            self.model = self._set_deeplabv3()
        else:
            raise RuntimeError('Please prepare {} model in modules/ .'.format(self.network))


    def process(self, img):
        # semantic segmentation
        semseg_map = self.model(img)

        return semseg_map


    def _set_deeplabv3(self):
        weights = os.path.join(os.getcwd(),
                'modules/deeplabv3/pretrained/model_13_2_2_2_epoch_580.pth')
        resnet_root = os.path.join(os.getcwd(),
            'modules/deeplabv3/pretrained/resnet')
        model = DeepLabV3(self.resnet, resnet_root, self.device)
        param = torch.load(weights, map_location=self.device)
        model.load_state_dict(param)
        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

        return model
