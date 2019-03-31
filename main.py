import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Config, Debugger, create_dir, pmd_mode_change
from config import get_config


def main():
    path = os.path.join(os.getcwd(), 'data/cityscapes/frankfurt')
    impath = os.path.join(path, 'input', 'frankfurt_000001_041664_leftImg8bit.png')

    # set configs
    config = Config(get_config(path))

    # create directories
    if config.mode == 'img':
        path = create_dir(impath)
    else:
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
    main()
