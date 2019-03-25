import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Config, Debugger


def test(path, config):
    config = Config(config)

    # setup environment
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # define the model
    model = GANonymizer(config, device)

    # model prediction
    inpath = os.path.join(path, 'input')
    files = os.listdir(inpath)
    files = [os.path.join(inpath, f) for f in files if os.path.isfile(os.path.join(inpath, f))]
    for file in files:
        model.predict(file)


if __name__ == '__main__':
    path = 'data/exp/cityscapes_addset'
    config = {
            # execution setting
            'checkpoint': os.path.join(path, 'ckpt'),
            'output': os.path.join(path, 'output'),

            # resize
            'resize_factor': 1,
        
            # mode (choose in ['pass', 'save', 'exec', 'debug', 'none'])
            'main_mode': 'save',
            'semseg_mode': 'save',
            'mask_mode': 'save',
            'shadow_mode': 'save',
            'divide_mode': 'save',
            'inpaint_mode': 'save',

            # segmentation
            'semseg': 'DeepLabV3',
            'resnet': 18,

            # mask
            'mask': 'entire',
            'expand_width': 6,
            # separate
            'crop_rate': 0.5,

            # shadow detection
            'obj_sml_thresh': 1e-3, # this param is also used in the pmd
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
            'sigma': 2 # for canny edge detection
            }
    
    test(path, config)

