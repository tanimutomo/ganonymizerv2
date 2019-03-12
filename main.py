import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Debugger, set_networks, labels


def main(img, config, main=False, debug=False):
    # setup environment
    debugger = Debugger(main, debug)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model prediction
    model = GANonymizer(config, device, labels, debugger)
    model.predict(img)


if __name__ == '__main__':
    img = os.path.join(os.getcwd(), 'data/exp/cityscapes_testset/ex_01.png')
    config = {
            # checkpoint
            'checkpoint': 'data/exp/cityscapes_testset',
        
            # mode
            'semseg_mode': 'save',
            'mask_mode': 'save',
            'inpaint_mode': 'save',
        
            # segmentation
            'semseg': 'DeepLabV3',
            'resnet': 18,

            # mask
            'mask': 'separate',

            # inpaint
            'inpaint': 'EdgeConnect',
            'checkpoints_path': 'modules/edge_connect/checkpoints',
            'sigma': 2
            }

    main(img, config, main=False, debug=False, save=False)

