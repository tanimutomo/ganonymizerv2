import os
import cv2
import torch


if os.path.basename(os.getcwd()) in ['src', 'app']:
    from modules.ganonymizer import GANonymizer
    from modules.utils import Config, Debugger, create_dir, \
            pmd_mode_change, load_img, demo_config, demo_resize
    from config import get_config
else:
    from .modules.ganonymizer import GANonymizer
    from .modules.utils import Config, Debugger, create_dir, \
            pmd_mode_change, load_img, demo_config, demo_resize
    from .config import get_config


def main(mode, data_root, filename=None):
    # raise argument error
    if filename:
        assert mode == 'img' or mode == 'demo'
        filepath = os.path.join(data_root, 'input', filename)
    else:
        assert mode == 'pmd' or mode == 'dir'

    # set configs
    config = Config(get_config(data_root))
    config.mode = mode

    # set moduels root path
    if os.path.basename(os.getcwd()) in ['src', 'app']:
        config.mod_root = os.path.join(os.getcwd(), 'modules')
    elif os.path.basename(os.getcwd()) == 'ganonymizerv2':
        config.mod_root = os.path.join(os.getcwd(), 'src', 'modules')

    # create directories
    if config.mode == 'img' or config.mode == 'demo':
        filepath = create_dir(filepath)
    else:
        dirpath = create_dir(data_root)

    # setup environment
    device = torch.device('cuda:{}'.format(config.cuda)
            if torch.cuda.is_available() else 'cpu')

    # if the mode is 'demo', all modules mode is changed 'exec'.
    if config.mode == 'demo':
        config = demo_config(config)

    # define the model
    model = GANonymizer(config, device)

    if config.mode == 'img' or config.mode == 'demo':
        print('Loading "{}"'.format(filepath)) 
        img, fname, fext = load_img(filepath)
        if config.mode == 'demo':
            img = demo_resize(img)
        config.fname = fname
        config.fext = fext
        # model prediction
        model.predict(img)

    elif config.mode == 'dir':
        inpath = os.path.join(dirpath, 'input')
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
        inpath = os.path.join(dirpath, 'input')
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
    # mode should be choosen from ['img', 'dir', 'pmd', 'demo']
    mode = 'img'
    data_root = os.path.join(os.getcwd(), 'data/exp/cityscapes_testset')
    filename = 'ex_01.png'
    main(mode, data_root, filename)

