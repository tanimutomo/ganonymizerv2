import cv2
import numpy as np
import os
import random
import skimage
import torch
import torchvision

from PIL import Image
from skimage.feature import canny
from torchvision import transforms

from .models import EdgeModel, InpaintingModel
from .config import Config

class SimpleEdgeConnect():
    def __init__(self, checkpoints_path, sigma, device, debugger):
        self.debugger = debugger
        self.device = device
        self.sigma = sigma
        self.checkpoints_path = checkpoints_path

        path = os.path.join(checkpoints_path, 'places2')
        config_path = os.path.join(path, 'config.yml')
        config = Config(config_path)

        # Model setting and forward
        # build the model and initialize
        self.edge_model = EdgeModel(config).to(device)
        self.inpaint_model = InpaintingModel(config).to(device)
        self.edge_model.load()
        self.inpaint_model.load()
        self.edge_model.eval()
        self.inpaint_model.eval()

        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)

        # initialize random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

        # transforms
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

    def inpaint(self, img, mask):
        gray = self.to_tensor(self.to_pil(gray).convert('L'))
        edge = self._get_edge(gray, mask)
        img, gray, mask, edge = self._cuda_tensor(img, gray, mask, edge)
        for item, name in [(img, 'img'), (mask, 'mask'), (gray, 'gray'), (edge, 'edge')]:
            self.debugger.matrix(item, name)

        # inpaint with edge model / joint model
        out_edge = self.edge_model(gray, edge, mask).detach()
        out = self.inpaint_model(img, edge, mask)
        out_merged = (out * mask) + (img * (1 - mask))
        return out_merged, out_edge, edge

    def _get_edge(self, gray, mask):
        gray = np.array(self.to_pil(gray)).astype(np.uint8)
        mask = np.array(self.to_pil(mask)).astype(np.uint8)
        mask = (1 - mask / 255).astype(np.bool)
        edge = canny(gray, sigma=self.sigma, mask=mask)
        return torch.from_numpy(edge).to(torch.float32)

    def _cuda_tensor(self, *args):
        items = list()
        for item in args:
            if item.ndim == 2:
                item = torch.unsqueeze(item, 0)
            items.append(item.unsqueeze(0).to(self.device))
        return items
