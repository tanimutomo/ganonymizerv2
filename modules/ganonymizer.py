import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from .semantic_segmenter import SemanticSegmenter
from .mask_creater import MaskCreater
from .shadow_detecter import ShadowDetecter
from .mask_divider import MaskDivider
from .inpainter import Inpainter
from .utils import Debugger


class GANonymizer:
    def __init__(self, config, device):
        self.config = config
        self.devide = device

        self.debugger = Debugger(config.main_mode, save_dir=config.checkpoint)
        self.to_tensor = ToTensor()

        self.semseger = SemanticSegmenter(config, device)
        self.mask_creater = MaskCreater(config)
        self.shadow_detecter = ShadowDetecter(config)
        self.inpainter = Inpainter(config, device)
        self.mask_divider = MaskDivider(config, self.inpainter)

    
    def predict(self, img_path):
        # loading input image
        img = self._load_img(img_path)

        # semantic segmentation for detecting dynamic objects and creating mask
        segmap = self._semseg(img)

        # use single entire mask (default)
        if self.config.mask == 'entire':

            # get object mask and shadow mask and combine them
            omask = self._object_mask(img, segmap)
            smask = self._detect_shadow(img, omask)
            mask = self._combine_masks(img, omask, smask)

            # Psuedo Mask Division
            divimg, divmask = self._divide_mask(img, mask)

            # image and edge inpainting
            out, _ = self._inpaint(divimg, divmask)

            # save output image by PIL
            out = Image.fromarray(out)
            out.save('./data/exp/cityscapes_testset/{}_out.{}'.format(
                self.fname, self.fext))

        # use separated mask for inpainting (this mode is failed)
        elif self.config.mask == 'separate':
            inputs = self._separated_mask(img, segmap)
            inpainteds = []
            for input in inputs:
                if input['area'] <= 400:
                    inpainted = cv2.inpaint(input['img'], input['mask'], 3, cv2.INPAINT_NS)
                else:
                    inpainted, inpainted_edge = self._inpaint(input['img'], input['mask'])
                inpainteds.append(inpainted)
            
            out = self._integrate_outputs(img, inputs, inpainteds)
            self.debugger.img(out, 'Final Output')
            out = Image.fromarray(out)
            out.save('./data/exp/cityscapes_testset/{}_sep_out_resized_{}.{}'.format(
                self.fname, self.config.resize_factor, self.fext))


    def _load_img(self, img_path):
        # Loading input image
        print('===== Loading Image =====')
        self.fname, self.fext = img_path.split('/')[-1].split('.')
        img = Image.open(img_path)
        img = img.resize((int(img.width / self.config.resize_factor),
            int(img.height / self.config.resize_factor)))
        img = np.array(img)

        # visualization
        self.debugger.img(img, 'Input Image')

        return img


    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')

        segmap_path = os.path.join(self.config.checkpoint, self.fname + '_segmap.' + 'pkl')
        if self.config.semseg_mode in ['exec', 'debug']:
            semseg_map = self.semseger.process(img)
        elif self.config.semseg_mode is 'pass':
            with open(segmap_path, mode='rb') as f:
                semseg_map = pickle.load(f)
        elif self.config.semseg_mode is 'save':
            semseg_map = self.semseger.process(img)
            with open(segmap_path, mode='wb') as f:
                pickle.dump(semseg_map, f)

        # visualization
        self.debugger.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')
        self.debugger.imsave(semseg_map, self.fname + '_semsegmap.' + self.fext)

        return semseg_map

    
    def _combine_masks(self, img, omask, smask):
        # check omask and smak shape
        assert omask.shape == smask.shape

        # combine the object mask and the shadow mask
        mask = np.where(omask + smask > 0, 255, 0).astype(np.uint8)
        self.debugger.img(mask, 'Mask (Object Mask + Shadow Mask)', gray=True)
        self.debugger.imsave(mask, self.fname + '_mask.' + self.fext)

        # visualization the mask overlayed image
        mask3c = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
        overlay = (img * 0.7 + mask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_mask_overlayed.' + self.fext)

        # image with mask
        mask3c = np.stack([mask for _ in range(3)], axis=-1)
        img_with_mask = np.where(mask3c==255, mask3c, img).astype(np.uint8)
        self.debugger.img(img_with_mask, 'Image with Mask')
        self.debugger.imsave(img_with_mask, self.fname + '_img_with_mask.' + self.fext)

        return mask


    def _object_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        omask_path = os.path.join(self.config.checkpoint, self.fname + '_omask.' + 'pkl')

        if self.config.mask_mode in ['exec', 'debug']:
            omask = self.mask_creater.mask(img, semseg_map)
        elif self.config.mask_mode is 'pass':
            with open(omask_path, mode='rb') as f:
                omask = pickle.load(f)
        elif self.config.mask_mode is 'save':
            omask = self.mask_creater.mask(img, semseg_map)
            with open(omask_path, mode='wb') as f:
                pickle.dump(omask, f)

        # visualization
        self.debugger.img(omask, 'Object Mask', gray=True)
        self.debugger.imsave(omask, self.fname + '_omask.' + self.fext)

        # visualize the mask overlayed image
        omask3c = np.stack([omask, np.zeros_like(omask), np.zeros_like(omask)], axis=-1)
        overlay = (img * 0.7 + omask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Object Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_omask_overlayed.' + self.fext)

        return omask


    def _separated_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Create separated inputs =====')
        sep_inputs_path = os.path.join(self.config.checkpoint, self.fname + '_sep_inputs.' + 'pkl')

        if self.config.mask_mode in ['exec', 'debug']:
            inputs = self.shadow_detecter.separated_mask(img, semseg_map, 
                    self.config.crop_rate)
        elif self.config.mask_mode is 'pass':
            with open(sep_inputs_path, mode='rb') as f:
                inputs = pickle.load(f)
        elif self.config.mask_mode is 'save':
            inputs = self.shadow_detecter.separated_mask(img, semseg_map, 
                    self.config.crop_rate)
            with open(sep_inputs_path, mode='wb') as f:
                pickle.dump(inputs, f)

        return inputs


    def _detect_shadow(self, img, mask):
        # shadow detection
        print('===== Shadow Detection =====')
        smask_path = os.path.join(self.config.checkpoint, self.fname + '_smask.' + 'pkl')

        if self.config.shadow_mode in ['exec', 'debug']:
            smask = self.shadow_detecter.detect(img, mask)
        elif self.config.shadow_mode is 'pass':
            with open(smask_path, mode='rb') as f:
                smask = pickle.load(f)
        elif self.config.shadow_mode is 'save':
            smask = self.shadow_detecter.detect(img, mask)
            with open(smask_path, mode='wb') as f:
                pickle.dump(smask, f)
        elif self.config.shadow_mode is 'none':
            smask = np.zeros_like(mask)

        # visualization
        self.debugger.img(smask, 'Shadow Mask', gray=True)
        self.debugger.imsave(smask, self.fname + '_smask.' + self.fext)

        # visualize the mask overlayed image
        smask3c = np.stack([smask, np.zeros_like(smask), np.zeros_like(smask)], axis=-1)
        overlay = (img * 0.7 + smask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Shadow Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_smask_overlayed.' + self.fext)

        return smask


    def _divide_mask(self, img, mask):
        # pseudo mask division
        print('===== Pseudo Mask Division =====')
        divimg, divmask = self.mask_divider.divide(img, mask, self.fname, self.fext)

        divimg_path = os.path.join(self.config.checkpoint, self.fname + '_divimg.' + 'pkl')
        divmask_path = os.path.join(self.config.checkpoint, self.fname + '_divmask.' + 'pkl')

        if self.config.divide_mode in ['exec', 'debug']:
            divimg, divmask = self.mask_divider.divide(img, mask, self.fname, self.fext)
        elif self.config.divide_mode is 'pass':
            with open(divimg_path, mode='rb') as f:
                divimg = pickle.load(f)
            with open(divmask_path, mode='rb') as f:
                divmask = pickle.load(f)
        elif self.config.divide_mode is 'save':
            divimg, divmask = self.mask_divider.divide(img, mask, self.fname, self.fext)
            with open(divimg_path, mode='wb') as f:
                pickle.dump(divimg, f)
            with open(divmask_path, mode='wb') as f:
                pickle.dump(divmask, f)
        elif self.config.divide_mode is 'none':
            divimg = img
            divmask = mask

        # visualization
        self.debugger.img(divimg, 'Divided Image')
        self.debugger.imsave(divimg, self.fname + '_divimg.' + self.fext)
        self.debugger.img(divmask, 'Divided Mask', gray=True)
        self.debugger.imsave(divmask, self.fname + '_divmask.' + self.fext)

        return divimg, divmask


    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting and Edge Inpainting =====')
        inpaint_path = os.path.join(self.config.checkpoint, self.fname + '_inpaint.' + 'pkl')
        inpaint_edge_path = os.path.join(self.config.checkpoint, self.fname + '_inpaint_edge.' + 'pkl')
        edge_path = os.path.join(self.config.checkpoint, self.fname + '_edge.' + 'pkl')

        if self.config.inpaint_mode in ['exec', 'debug']:
            inpainted, inpainted_edge, edge = self.inpainter.inpaint(img, mask)
        elif self.config.inpaint_mode is 'pass':
            with open(inpaint_path, mode='rb') as f:
                inpainted = pickle.load(f)
            with open(inpaint_edge_path, mode='rb') as f:
                inpainted_edge = pickle.load(f)
            with open(edge_path, mode='rb') as f:
                edge = pickle.load(f)
        elif self.config.inpaint_mode is 'save':
            inpainted, inpainted_edge, edge = self.inpainter.inpaint(img, mask)
            with open(inpaint_path, mode='wb') as f:
                pickle.dump(inpainted, f)
            with open(inpaint_edge_path, mode='wb') as f:
                pickle.dump(inpainted_edge, f)
            with open(edge_path, mode='wb') as f:
                pickle.dump(edge, f)

        # visualization
        self.debugger.img(edge, 'Edge', gray=True)
        self.debugger.imsave(edge, self.fname + '_edge.' + self.fext)
        self.debugger.img(inpainted_edge, 'Inpainted Edge', gray=True)
        self.debugger.imsave(inpainted_edge, self.fname + '_inpainted_edge.' + self.fext)
        self.debugger.img(inpainted, 'Inpainted Image')
        self.debugger.imsave(inpainted, self.fname + '_inpainted.' + self.fext)

        return inpainted, inpainted_edge


    def _integrate_outputs(self, img, inputs, outputs):
        self.debugger.img(img, 'original image')
        for input, output in zip(inputs, outputs):
            box = input['box']
            origin_wh = (box[2] - box[0], box[3] - box[1])
            output = cv2.resize(output, origin_wh)
            mask_part = cv2.resize(input['mask'], origin_wh)
            self.debugger.img(output, 'output')

            out = np.zeros(img.shape, dtype=np.uint8)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            out[box[1]:box[3], box[0]:box[2], :] = output
            self.debugger.img(out, 'out')
            mask[box[1]:box[3], box[0]:box[2]] = mask_part
            
            mask = np.stack([mask for _ in range(3)], axis=-1)
            img = np.where(mask==255, out, img)
            self.debugger.img(img, 'img')

        return img


