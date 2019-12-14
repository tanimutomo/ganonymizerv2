import copy
import cv2
import numpy as np
import os
import torch


if os.path.basename(os.getcwd()) in ['src', 'app']:
    from modules.ganonymizer import GANonymizer
    from modules.utils import Config, Debugger, create_dir, \
            pmd_mode_change, load_img, demo_config, demo_resize, load_video, video_writer
    from config import get_config
else:
    from .modules.ganonymizer import GANonymizer
    from .modules.utils import Config, Debugger, create_dir, \
            pmd_mode_change, load_img, demo_config, demo_resize, load_video
    from .config import get_config


def main(mode, data_root, filename=None):
    # raise argument error
    if filename:
        assert mode == 'img' or mode == 'demo' or mode == "video"
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
    if config.mode == 'demo' or config.mode == "video":
        config = demo_config(config)

    # define the model
    model = GANonymizer(config, device)

    if config.mode == 'img' or config.mode == 'demo':
        print('Loading "{}"'.format(filepath)) 
        img, fname, fext = load_img(filepath)
        if config.mode == 'demo':
            img = demo_resize(np.array(img))
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

    elif config.mode == "video":
        print("Loading '{}'".format(filepath))
        count = 1
        # Load the video
        fname, cap, origin_fps, frames, width, height = load_video(filepath)
        writer = video_writer(filepath, origin_fps, width, height*2)
        config.fname, config.fext = "0", "0"

        while(cap.isOpened()):
            print('')
            ret, frame = cap.read()
            if ret:
                print('-----------------------------------------------------')
                print('[INFO] Count: {}/{}'.format(count, frames))

                # process
                img = copy.deepcopy(frame)
                output = model.predict(img)
                concat = np.concatenate([frame, output], axis=0)
                writer.write(concat)
                count += 1

            else:
                break

        # Stop video process
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # mode should be choosen from ['img', 'dir', 'pmd', 'demo']
    mode = "video"
    data_root = os.path.join(os.getcwd(), "data/video")
    filename = "noon_half_vshort.avi"
    main(mode, data_root, filename)

