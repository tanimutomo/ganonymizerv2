import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Debugger, set_networks, labels


def main(img, config, main=False, debug=False, save=False):
    # setup environment
    debugger = Debugger(main, debug, save, save_dir=config['checkpoint'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model prediction
    model = GANonymizer(config, device, labels, debugger)
    model.predict(img)


if __name__ == '__main__':
    img = os.path.join(os.getcwd(), 'data/exp/cityscapes_testset/ex_01.png')
    config = {
            # checkpoint
            'checkpoint': 'data/exp/cityscapes_testset',

            # resize
            'resize_factor': 1,
        
            # mode
            'semseg_mode': 'save',
            'mask_mode': 'save',
            'shadow_mode': 'save',
            'inpaint_mode': 'save',
        
            # segmentation
            'semseg': 'DeepLabV3',
            'resnet': 18,

            # mask
            'mask': 'entire',
            # separate
            'crop_rate': 0.5,

            # shadow detection
            'obj_sml_thresh': 1e-3,
            'obj_high_thresh': 0.2,
            'superpixel': 'quickshift',
            'shadow_high_thresh': 0.01,
            'alw_range_max': 30,
            'find_iteration': 3,
            'ss_score_thresh': 4,
            'sc_color_thresh': 1.5,

            # inpaint
            'inpaint': 'EdgeConnect',
            'inpaint_ckpt': 'modules/edge_connect/checkpoints',
            'sigma': 2
            }

    main(img, config, main=False, debug=False, save=False)

