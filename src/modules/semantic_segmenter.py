import os
import torch


from .deeplabv3.model.deeplabv3 import DeepLabV3

class SemanticSegmenter():
    def __init__(self, opt):
        if opt.semseg == 'DeepLabV3':
            self.model = self._set_deeplabv3(opt)
        else:
            raise NotImplementedError('Please prepare {} model in modules/ .'.format(opt.semseg))

    def predict(self, img):
        # semantic segmentation
        label_map = self.model(img)
        return label_map

    def _set_deeplabv3(self, opt):
        weights = 'modules/deeplabv3/pretrained/model_13_2_2_2_epoch_580.pth'
        resnet_root = 'modules/deeplabv3/pretrained/resnet'
        model = DeepLabV3(opt.resnet, resnet_root, opt.device)
        param = torch.load(weights, map_location=opt.device)
        model.load_state_dict(param)
        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

        return model
