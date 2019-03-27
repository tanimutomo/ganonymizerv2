import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Config, Debugger, create_dir, pmd_mode_change


def main(path, config):
    config = Config(config)

    # create directories
    path = create_dir(path)

    # setup environment
    device = torch.device('cuda:{}'.format(config.cuda)
            if torch.cuda.is_available() else 'cpu')

    # define the model
    model = GANonymizer(config, device)

    if config.mode == 'img':
        print('Loading "{}"'.format(path)) 
        # model prediction
        model.predict(path)

    elif config.mode == 'dir':
        inpath = os.path.join(path, 'input')
        files = os.listdir(inpath)
        files = [os.path.join(inpath, f) for f in files 
                if os.path.isfile(os.path.join(inpath, f)) and f[0] != '.']
        for f in files:
            print('Loading "{}"'.format(f)) 
            model.predict(f)

    elif config.mode == 'pmd':
        inpath = os.path.join(path, 'input')
        files = os.listdir(inpath)
        files = [os.path.join(inpath, f) for f in files 
                if os.path.isfile(os.path.join(inpath, f)) and f[0] != '.']
        for f in files:
            print('Loading "{}"'.format(f)) 

            # with pmd
            config = pmd_mode_change(config, 'on')
            model.reload_config(config)
            out_on = model.predict(f)
            
            # without pmd
            config = pmd_mode_change(config, 'off')
            model.reload_config(config)
            out_off = model.predict(f)

            # calcurate psnr and ssim



if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'data/exp/ablation_images')
    impath = os.path.join(path, 'input', 'frankfurt_000000_000576_leftImg8bit.png')
    config = {
            # execution setting
            # mode should be choosen from ['img', 'dir', 'pmd']
            'mode': 'pmd',
            'checkpoint': os.path.join(path, 'ckpt'),
            'output': os.path.join(path, 'output'),
            'cuda': 1,

            # *_mode (choose in ['pass', 'save', 'exec', 'debug', 'none'])
            'main_mode': 'exec',
            'semseg_mode': 'pass',
            'mask_mode': 'pass',
            'split_mode': 'pass',
            'shadow_mode': 'pass',
            'random_mode': 'pass', # evaluate pmd
            'divide_mode': 'pass',
            'inpaint_mode': 'pass',

            # evaluate pmd
            'eval_pmd_path': os.path.join(path, 'pmd'),
            'rmask_min': 100,
            'rmask_max': 400,

            # resize
            'resize_factor': 1,
        
            # segmentation
            'semseg': 'DeepLabV3',
            'resnet': 18,

            # mask
            'mask': 'entire',
            'expand_width': 6,
            # separate
            'crop_rate': 0.5,
            
            # separate mask to each object
            'fill_hole': 'later',
            'obj_sml_thresh': 1e-3, # this param is also used in shadow detection
            'obj_sep_thresh': 1/3,

            # shadow detection
            'obj_high_thresh': 0.2,
            'superpixel': 'quickshift',
            'shadow_high_thresh': 0.01,
            'alw_range_max': 15,
            'find_iteration': 3,
            'ss_score_thresh': 4,
            'sc_color_thresh': 1.5,

            # pseudo mask division
            'obj_wh_thresh': 120,
            'obj_density_thresh': 0.4,
            'line_width_div': 8,

            # inpaint
            'inpaint': 'EdgeConnect',
            'inpaint_ckpt': 'modules/edge_connect/checkpoints',
            'sigma': 1 # for canny edge detection
            }
    
    main(path, config)
