import os
import cv2
import random
import numpy as np
import torch
from src.config import Config
from src.edge_connect_edit import SimpleEdgeConnect
from src.models import EdgeModel, InpaintingModel
from src.utils import Debugger


def main(mode=None):
    # configuration setting
    path = './checkpoints/places2'
    config_path = os.path.join(path, 'config.yml')
    config = Config(config_path)
    config.MODE = 2
    config.MODEL = 3

    # for debug
    debugger = Debugger(True, False)

    # cuda visble devices
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Model setting and forward
    # build the model and initialize
    print(device)
    edge_model = EdgeModel(config).to(device)
    inpaint_model = InpaintingModel(config).to(device)
    edge_model.load()
    inpaint_model.load()

    sigma = 2
    model = SimpleEdgeConnect(device, edge_model, inpaint_model, sigma, debugger)

    print('\nstart testing...\n')
    model.process('./checkpoints/exp/inputs/places2_01.png')


if __name__ == "__main__":
    main()
