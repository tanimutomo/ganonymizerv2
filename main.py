import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Config, Debugger


def main(img, config):
    config = Config(config)

    # setup environment
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # model prediction
    model = GANonymizer(config, device)
    model.predict(img)


if __name__ == '__main__':
    img = os.path.join(os.getcwd(), 'data/exp/cityscapes_testset/ex_01.png')
    config = {
            # execution setting
            'checkpoint': 'data/exp/cityscapes_testset',

            # resize
            'resize_factor': 1,
        
            # mode (choose in ['pass', 'exec', 'debug', 'save'])
            'main_mode': 'exec',
            'semseg_mode': 'pass',
            'mask_mode': 'pass',
            'shadow_mode': 'pass',
            'divide_mode': 'exec',
            'inpaint_mode': 'pass',

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
            'alw_range_max': 15,
            'find_iteration': 3,
            'ss_score_thresh': 4,
            'sc_color_thresh': 1.5,

            # pseudo mask division
            'obj_wh_thresh': 120,
            'obj_density_thresh': 0.6,

            # inpaint
            'inpaint': 'EdgeConnect',
            'inpaint_ckpt': 'modules/edge_connect/checkpoints',
            'sigma': 2
            }
    
    main(img, config)

