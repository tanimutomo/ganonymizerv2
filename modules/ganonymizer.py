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
from .utils import Debugger, expand_mask


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
            out = self._inpaint(divimg, divmask)

            # save output image by PIL
            out = Image.fromarray(out)
            shadow = 'off' if self.config.shadow_mode is 'none' else 'on'
            pmd = 'off' if self.config.divide_mode is 'none' else 'on'
            out.save(os.path.join(self.config.output,
                '{}_out_expanded_{}_shadow_{}_pmd_{}.{}'.format(
                self.fname, self.config.expand_width, shadow, pmd, self.fext)))

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


    def _combine_masks(self, img, omask, smask):
        # check omask and smak shape
        assert omask.shape == smask.shape

        # combine the object mask and the shadow mask
        mask = np.where(omask + smask > 0, 255, 0).astype(np.uint8)
        mask = expand_mask(mask, self.config.expand_width)
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


    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')
        semseg_map = self._exec_module(self.config.semseg_mode, 'segmap',
                self.semseger.process, img)

        return semseg_map

    
    def _object_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        omask = self._exec_module(self.config.mask_mode, 'omask',
                self.mask_creater.mask, img, semseg_map)

        # visualize the mask overlayed image
        omask3c = np.stack([omask, np.zeros_like(omask), np.zeros_like(omask)], axis=-1)
        overlay = (img * 0.7 + omask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Object Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_omask_overlayed.' + self.fext)

        return omask


    def _detect_shadow(self, img, mask):
        # shadow detection
        print('===== Shadow Detection =====')
        if self.config.shadow_mode is 'none':
            smask = np.zeros_like(mask).astype(np.uint8)
        else:
            smask = self._exec_module(self.config.shadow_mode, 'smask',
                    self.shadow_detecter.detect, img, mask)

        # visualize the mask overlayed image
        smask3c = np.stack([smask, np.zeros_like(smask), np.zeros_like(smask)], axis=-1)
        overlay = (img * 0.7 + smask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Shadow Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_smask_overlayed.' + self.fext)

        return smask


    def _divide_mask(self, img, mask):
        # pseudo mask division
        print('===== Pseudo Mask Division =====')
        if self.config.divide_mode is 'none':
            divimg = img
            divmask = mask
        else:
            divimg, divmask = self._exec_module(self.config.divide_mode, ['divimg', 'divmask'],
                    self.mask_divider.divide, img, mask, self.fname, self.fext)

        return divimg, divmask


    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting and Edge Inpainting =====')
        inpainted, inpainted_edge, edge = self._exec_module(self.config.inpaint_mode,
                ['inpaint', 'inpaint_edge', 'edge'], self.inpainter.inpaint, img, mask)

        return inpainted


    def _separated_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Create separated inputs =====')
        inputs = self._exec_module(self.config.mask_mode, 'sep_inputs',
                self.mask_creater.separated_mask, img, semseg_map)

        return inputs


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


    def _exec_module(self, mode, names, func, *args):
        if type(names) == str:
            path = os.path.join(self.config.checkpoint, self.fname + '_{}.'.format(names) + 'pkl')
        else:
            paths = []
            for name in names:
                paths.append(os.path.join(self.config.checkpoint,
                    self.fname + '_{}.'.format(name) + 'pkl'))

        if mode in ['exec', 'debug']:
            if type(names) == str:
                result = func(*args)
            else:
                results = list(func(*args))
        elif mode is 'pass':
            if type(names) == str:
                with open(path, mode='rb') as f:
                    result = pickle.load(f)
            else:
                results = []
                for path in paths:
                    with open(path, mode='rb') as f:
                        results.append(pickle.load(f))
        elif mode is 'save':
            if type(names) == str:
                result = func(*args)
                with open(path, mode='wb') as f:
                    pickle.dump(result, f)
            else:
                results = list(func(*args))
                for res, path in zip(results, paths):
                    with open(path, mode='wb') as f:
                        pickle.dump(res, f)

        # visualization
        if type(names) == str:
            self.debugger.img(result, names)
            self.debugger.imsave(result, self.fname + '_{}.'.format(names) + self.fext)
            return result

        else:
            for res, name in zip(results, names):
                self.debugger.img(res, name)
                self.debugger.imsave(res, self.fname + '_{}.'.format(name) + self.fext)
            return results

