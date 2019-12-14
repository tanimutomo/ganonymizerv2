import os
import PIL

from PIL import Image

from modules.ganonymizer import GANonymizer
from options import get_options


def main(args=None):
    opt = get_options(args)

    print('Loading "{}"'.format(opt.input)) 
    img, opt.fname, opt.fext = load_img(opt.input)
    model = GANonymizer(opt)
    out = model(img)
    out.save(os.path.join(opt.log, 'output.png'))


def load_img(img_path):
    # Loading input image
    print('===== Loading Image =====')
    fname, fext = img_path.split('/')[-1].split('.')
    img = Image.open(img_path)
    return img, fname, fext


if __name__ == '__main__':
    main()
