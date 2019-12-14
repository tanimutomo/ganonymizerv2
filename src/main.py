import copy
import cv2
import numpy as np
import os

from modules.ganonymizer import GANonymizer
from modules.utils import load_img, load_video
from options import get_options


def main(mode, args=None):
    opt = get_options(args)
    opt.mode = mode

    if opt.mode == 'img':
        print('Loading "{}"'.format(opt.input)) 
        img, opt.fname, opt.fext = load_img(opt.input)
        model = GANonymizer(opt)
        out = model(img)
        out.save(os.path.join(opt.log, 'output.png'))

    elif opt.mode == "video":
        print("Loading '{}'".format(opt.input))
        count = 1
        # Load the video
        fname, cap, origin_fps, frames, width, height = load_video(opt.input)
        writer = video_writer(opt.input, origin_fps, width, height*2)
        opt.fname, opt.fext = "0", "0"
        model = GANonymizer(opt)

        while(cap.isOpened()):
            print('')
            ret, frame = cap.read()
            if ret:
                print('-----------------------------------------------------')
                print('[INFO] Count: {}/{}'.format(count, frames))

                # process
                img = copy.deepcopy(frame)
                output = model(img)
                output = np.array(output)
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
    mode = "img"
    main(mode)
