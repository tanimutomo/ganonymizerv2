import os
import cv2
import torch

from modules.model import GANonymizer
from modules.utils import Debugger, set_networks, labels


def main(img, config, debug=False, save=False):
    # setup environment
    debugger = Debugger(debug, save)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model prediction
    model = GANonymizer(config, device, labels, debugger)
    model.predict(img)


if __name__ == '__main__':
    img = os.path.join(os.getcwd(), 'data/examples/example_01.jpg')
    config = {
            'resnet': 18
            }

    main(img, config, debug=False, save=False)

