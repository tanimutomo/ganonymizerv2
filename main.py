import os
import cv2
import torch

from modules.model import GANonymizer
from modules.deeplabv3.model.deeplabv3 import DeepLabV3
from modules.utils import Debugger, set_networks
from modules.shadow_detecter import ShadowDetecter


def main(img, params, debug=False, save=False):
    # setup environment
    debugger = Debugger(debug, save)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup models
    semseger = set_networks(DeepLabV3, params['resnet'], device)
    shadow_detecter = ShadowDetecter(debugger)
    inpainter = None

    # model prediction
    model = GANonymizer(params, device, semseger, inpainter, shadow_detecter, debugger)
    model.predict(img)


if __name__ == '__main__':
    img = os.path.join(os.getcwd(), 'data/examples/example_01.jpg')
    params = {
            'resnet': 18
            }

    main(img, params, debug=False, save=False)

