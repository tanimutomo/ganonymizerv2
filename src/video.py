import copy
import cv2
import numpy as np
import os

from modules.ganonymizer import GANonymizer
from options import get_options


def main(args=None):
    opt = get_options(args)

    print("Loading '{}'".format(opt.input))
    count = 1
    # Load the video
    fname, cap, origin_fps, frames, width, height = load_video(opt.input)
    writer = video_writer(os.path.join(opt.log, 'output.avi'),
                          origin_fps, width, height*2)
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


def load_video(path):
    print('===== Loading Video =====')
    fname, fext = path.split('/')[-1].split('.')
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))
    return fname, cap, fps, frames, W, H
    

def video_writer(path, fps, width, height):
    print("Saved Video Path:", path)
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    return writer



if __name__ == '__main__':
    main()
