import os
import torch
import numpy as np
from PIL import Image
from skimage.feature import canny
from torchvision import transforms


class SimpleEdgeConnect():
    def __init__(self, device, edge_model, inpaint_model, sigma, debugger):
        self.debugger = debugger
        self.device = device
        self.edge_model = edge_model.eval()
        self.inpaint_model = inpaint_model.eval()
        self.sigma = sigma


    def process(self, path):
        # for dev
        img, gray, mask, edge = self._load_inputs(path, self.sigma)
        img, gray, mask, edge = self._cuda(img, gray, mask, edge)

        # inpaint with edge model / joint model
        edge = self.edge_model(gray, edge, mask).detach()
        output = self.inpaint_model(img, edge, mask)
        output_merged = (output * mask) + (img * (1 - mask))

        output = self._postprocess(output_merged)[0]

        self.debugger.imsave(output, os.path.join('./checkpoints/exp/outputs', path.split('/')[-1]))


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


