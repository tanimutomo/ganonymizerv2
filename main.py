import os
import cv2
import torch

from modules.ganonymizer import GANonymizer
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
            # segmentation
            'semseg': 'DeepLabV3',
            'resnet': 18,

            # inpaint
            'inpaint': 'EdgeConnect',
            'checkpoints_path': 'modules/edge_connect/checkpoints'
            }

    main(img, config, debug=False, save=False)

