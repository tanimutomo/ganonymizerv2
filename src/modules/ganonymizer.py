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
from .utils import Debugger, label_img_to_color, expand_mask


class GANonymizer:
    def __init__(self, opt):
        for k, v in opt.__dict__.items():
            setattr(self, k, v)
        self.debugger = Debugger(opt.main_mode, save_dir=opt.log)

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
        img = self.to_tensor(pil_img) # to torch.tensor in [0, 1]
        self.debugger.img(img, 'Input Image')

        # semantic segmentation for detecting dynamic objects and creating mask
        label_map = self._semseg(img)

        # get object mask
        mask = self._object_mask(img, label_map)

        # resize inputs
        img, mask = self._resize(img, mask)

        # image and edge inpainting
        out = self._inpaint(img, mask)

        return self.to_pil(out)

    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')
        label_map = self.ss.predict(img)

        vis, lc_img = label_img_to_color(label_map)
        self.debugger.img(vis, 'color semseg map')
        self.debugger.img(lc_img, 'label color map')
        return label_map

    def _object_mask(self, img, label_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        omask = self.mc.create_mask(label_map) # shape=(h, w) # dtype=torch.uint8

        # visualize the mask overlayed image
        omask3c = torch.stack([omask, torch.zeros_like(omask), torch.zeros_like(omask)], dim=0)
        img = (img * 255).to(torch.uint8)
        overlay = (img * 0.8 + omask3c * 0.2).to(torch.uint8)
        self.debugger.img(overlay, 'Object Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_omask_overlayed.' + self.fext)
        return omask

    def _resize(self, img, mask):
        # resize inputs
        if self.resize_factor is None:
            return img, mask

        new_h = img.shape[1] // self.resize_factor
        new_w = img.shape[2] // self.resize_factor

        img = self.to_pil(img).resize((new_w, new_h))
        mask = self.to_pil(mask).resize((new_w, new_h))
        return self.to_tensor(img), self.to_tensor(mask)

    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting =====')
        inpainted, inpainted_edge, edge = self.ii.inpaint(img, mask)
        return inpainted
