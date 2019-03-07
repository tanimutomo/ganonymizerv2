import os
import cv2
import random
import torch
import numpy as np
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


    def inpaint(self, img, mask):
        # img and mask is np.ndarray
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = self._get_edge(gray, mask)
        for item, name in [(img, 'img'), (mask, 'mask'), (gray, 'gray'), (edge, 'edge')]:
            if len(item.shape) == 3:
                self.debugger.img(item, name)
            else:
                self.debugger.img(item, name, gray=True)
        img, gray, mask, edge = self._cuda_tensor(img, gray, mask, edge)
        for item, name in [(img, 'img'), (mask, 'mask'), (gray, 'gray'), (edge, 'edge')]:
            self.debugger.matrix(item, name)

        # inpaint with edge model / joint model
        out_edge = self.edge_model(gray, edge, mask).detach()
        output = self.inpaint_model(img, edge, mask)
        output_merged = (output * mask) + (img * (1 - mask))

        out_edge = self._postprocess(out_edge)
        output = self._postprocess(output_merged)

        return output, out_edge


    def _get_edge(self, gray, mask):
        gray = (np.array(gray)).astype(np.uint8)
        mask = (1 - np.array(mask) / 255).astype(np.bool)
        edge = canny(gray, sigma=self.sigma, mask=mask)

        return edge * 255


    def _cuda_tensor(self, *args):
        items = []
        for item in args:
            item = torch.from_numpy(item / 255.0)
            if len(list(item.shape)) == 3:
                item = item.permute(2, 0, 1)
            else:
                item = item.unsqueeze(0)
            items.append(item.unsqueeze(0).to(
                torch.float32).to(self.device))
        return items


    def _postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1).squeeze()
        img = img.detach().cpu().numpy().astype(np.uint8)
        return img


    def _load_inputs_from_path(self, path, sigma):
        # path
        path_dir = path.split('/')[:-1]
        fname, fext = path.split('/')[-1].split('.')
        mask_path = os.path.join(*path_dir, fname + '_mask' + '.' + fext)

        # load images
        img = Image.open(path)
        gray = img.convert('L')
        mask = Image.open(mask_path)
        edge = self._get_edge(gray, mask, sigma)

        # convert to tensor from pil image
        to_tensor = transforms.ToTensor()
        img = to_tensor(img).unsqueeze(0).to(torch.float32)
        gray = to_tensor(gray).unsqueeze(0).to(torch.float32)
        mask = to_tensor(mask).unsqueeze(0).to(torch.float32)

        return img, gray, mask, edge


