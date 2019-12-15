import cv2
import numpy as np
import os
import pickle
import PIL
import torch
import torchvision

from PIL import Image
from torchvision import transforms

from .semantic_segmenter import SemanticSegmenter
from .mask_creater import MaskCreater
from .inpainter import ImageInpainter
from .utils import Debugger, label_img_to_color


class GANonymizer:
    def __init__(self, opt):
        for k, v in opt.__dict__.items():
            setattr(self, k, v)
        self.debugger = Debugger(opt.mode, save_dir=opt.inter_log)

        self.ss = SemanticSegmenter(opt)
        self.mc = MaskCreater(opt)
        self.ii = ImageInpainter(opt)

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, pil_img):
        """
        Args:
            pil_img : input image (PIL.Image)

        Returns:
            output image (PIL.Image)
        """
        img = self._preprocess(pil_img) # to torch.tensor in [0, 1]

        # semantic segmentation for detecting dynamic objects and creating mask
        label_map = self._semseg(img)

        # get object mask
        mask = self._create_mask(img, label_map)

        # image and edge inpainting
        out = self._inpaint(img, mask)

        return self.to_pil(out)

    def _preprocess(self, img):
        img = self.to_tensor(img)
        if self.resize_factor is None:
            return img
        new_h = int(img.shape[1] * self.resize_factor)
        new_w = int(img.shape[2] * self.resize_factor)
        new_h -= new_h % 4
        new_w -= new_w % 4
        img = self.to_pil(img).resize((new_w, new_h))
        return self.to_tensor(img)

    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')
        label_map = self.ss.predict(img)

        vis, lc_img = label_img_to_color(label_map)
        self.debugger.imsave(vis, 'color_semseg_map.png')
        self.debugger.imsave(lc_img, 'label_color_map.png')
        return label_map

    def _create_mask(self, img, label_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        mask = self.mc.create_mask(label_map) # shape=(h, w) # dtype=torch.uint8

        # visualize the mask overlayed image
        mask3c = torch.stack([mask, torch.zeros_like(mask), torch.zeros_like(mask)], dim=0)
        img = (img * 255).to(torch.uint8)
        overlay = (img * 0.8 + mask3c * 0.2).to(torch.uint8)
        self.debugger.imsave(overlay, 'mask_overlayed.png')
        return mask

    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting =====')
        inpainted, inpainted_edge, edge = self.ii.inpaint(img, mask)
        self.debugger.imsave(edge, 'edge.png')
        self.debugger.imsave(inpainted_edge, 'inpainted_edge.png')
        return inpainted
