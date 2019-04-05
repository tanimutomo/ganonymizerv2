import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
from modules.utils import Config, Debugger, create_dir, pmd_mode_change, load_img
from config import get_config


def main():
    path = os.path.join(os.getcwd(), 'data/exp/cityscapes_testset_1')
    impath = os.path.join(path, 'input', 'ex_01.png')

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
        img, fname, fext = load_img(path)
        config.fname = fname
        config.fext = fext
        # model prediction
        model.predict(img)

    elif config.mode == 'dir':
        inpath = os.path.join(path, 'input')
        files = os.listdir(inpath)
        files = [os.path.join(inpath, f) for f in files 
                if os.path.isfile(os.path.join(inpath, f)) and f[0] != '.']
        for f in files:
            print('Loading "{}"'.format(f)) 
            img, fname, fext = load_img(f)
            config.fname = fname
            config.fext = fext
            # model prediction
            model.predict(img)

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
            img, fname, fext = load_img(f)
            config.fname = fname
            config.fext = fext
            # model prediction
            out_on = model.predict(img)
            
            # without pmd
            config = pmd_mode_change(config, 'off')
            model.reload_config(config)
            img, fname, fext = load_img(f)
            config.fname = fname
            config.fext = fext
            # model prediction
            out_on = model.predict(img)



if __name__ == '__main__':
    main()
