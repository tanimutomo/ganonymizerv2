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
        print('device:', device)
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


    def inpaint(self, path):
        # for dev
        img, gray, mask, edge = self._load_inputs(path, self.sigma)
        img, gray, mask, edge = self._cuda(img, gray, mask, edge)

        # inpaint with edge model / joint model
        edge = self.edge_model(gray, edge, mask).detach()
        output = self.inpaint_model(img, edge, mask)
        output_merged = (output * mask) + (img * (1 - mask))

        output = self._postprocess(output_merged)[0]

        self.debugger.imsave(output, os.path.join(self.checkpoints_path, 'exp/outputs', path.split('/')[-1]))


    def _inputs(self, path):
        to_tensor = transforms.ToTensor()
        path_dir = path.split('/')[:-1]
        fname, fext = path.split('/')[-1].split('.')
        inputs = []
        for item in ['', '_gray', '_edge', '_mask']:
            path = os.path.join(*path_dir, fname + item + '.' + fext)
            img = Image.open(path)
            # img = img.resize((264, 264))
            img = to_tensor(img).unsqueeze(0)
            inputs.append(img.to(torch.float32))

        return inputs


    def _load_inputs(self, path, sigma):
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


    def _get_edge(self, gray, mask, sigma):
        gray = (np.array(gray) / 255).astype(np.bool)
        mask = (np.array(mask) / 255).astype(np.bool)
        edge = canny(gray, sigma=sigma, mask=mask).astype(np.float32)
        edge = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0)

        return edge


    def _cuda(self, *args):
        return (item.to(self.device) for item in args)


    def _postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


