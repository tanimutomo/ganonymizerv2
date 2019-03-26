import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image

from torchvision.transforms import ToTensor, ToPILImage

from .semantic_segmenter import SemanticSegmenter
from .mask_creater import MaskCreater
from .object_spliter import ObjectSpliter
from .shadow_detecter import ShadowDetecter
from .mask_divider import MaskDivider
from .inpainter import ImageInpainter
from .utils import Debugger, label_img_to_color, expand_mask, random_mask

class GANonymizer:
    def __init__(self, config, device):
        self.config = config
        self.devide = device
        self.debugger = Debugger(config.main_mode, save_dir=config.checkpoint)

        self.ss = SemanticSegmenter(config, device)
        self.mc = MaskCreater(config)
        self.op = ObjectSpliter(config)
        self.sd = ShadowDetecter(config)
        self.ii = ImageInpainter(config, device)
        self.md = MaskDivider(config, self.ii)
    
    def predict(self, img_path):
        # loading input image
        img = self._load_img(img_path)

        # semantic segmentation for detecting dynamic objects and creating mask
        segmap = self._semseg(img)

        # get object mask and shadow mask and combine them
        omask = self._object_mask(img, segmap)

        # get hole filled object mask and object labelmap
        omask, labelmap = self._split_object(omask)

        # detect shadow area and add shadow area to object mask
        smask, labelmap = self._detect_shadow(img, labelmap)
        mask, labelmap = self._combine_masks(img, omask, smask, labelmap)

        # evaluate the effect of PMD
        if self.config.evaluate_pmd:
            mask, labelmap = random_mask(mask, labelmap)

        # Psuedo Mask Division
        divimg, divmask = self._divide_mask(img, mask, labelmap)

        # resize inputs
        divimg = self._resize(divimg)
        divmask = self._resize(divmask)
        divmask = np.where(divmask > 127, 255, 0).astype(np.uint8)

        # image and edge inpainting
        out = self._inpaint(divimg, divmask)

        # save output image by PIL
        outimg = Image.fromarray(out)
        shadow = 'off' if self.config.shadow_mode is 'none' else 'on'
        pmd = 'off' if self.config.divide_mode is 'none' else 'on'
        outimg.save(os.path.join(self.config.output,
            '{}_out_expanded_{}_shadow_{}_pmd_{}.{}'.format(
            self.fname, self.config.expand_width, shadow, pmd, self.fext)))

        return out


    def _load_img(self, img_path):
        # Loading input image
        print('===== Loading Image =====')
        self.fname, self.fext = img_path.split('/')[-1].split('.')
        img = Image.open(img_path)
        img = np.array(img)

        # visualization
        self.debugger.img(img, 'Input Image')

        return img

    def _resize(self, img):
        # resize inputs
        new_h = int(img.shape[0] / self.config.resize_factor)
        new_w = int(img.shape[1] / self.config.resize_factor)
        out = cv2.resize(img, (new_w, new_h)).astype(np.uint8)
        return out

    def _combine_masks(self, img, omask, smask, labelmap):
        # check omask and smak shape
        assert omask.shape == smask.shape

        # combine the object mask and the shadow mask
        mask = np.where(omask + smask > 0, 255, 0).astype(np.uint8)
        mask = expand_mask(mask, self.config.expand_width)
        self.debugger.img(mask, 'Mask (Object Mask + Shadow Mask)')
        self.debugger.imsave(mask, self.fname + '_mask.' + self.fext)

        # expand labelmap
        for label in range(1, np.max(labelmap)+1):
            objmap = np.where(labelmap == label, 1, 0)
            objmap = expand_mask(objmap, self.config.expand_width)
            objmap = (objmap / 255 * label).astype(np.int32)
            labelmap = np.where(labelmap == 0, objmap, labelmap)
        self.debugger.img(labelmap, 'Expanded LabelMap')

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

        return mask, labelmap

    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')
        if self.config.semseg_mode is 'none':
            raise RuntimeError('semseg_mode should not be "none", if you want to execute entire GANonymizerV2')
        else:
            semseg_map = self._exec_module(self.config.semseg_mode,
                                           'segmap',
                                           self.ss.predict,
                                           img)

        vis, lc_img = label_img_to_color(semseg_map)
        self.debugger.img(vis, 'color semseg map')
        self.debugger.img(lc_img, 'label color map')

        return semseg_map
    
    def _object_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        if self.config.mask_mode is 'none':
            raise RuntimeError('mask_mode should not be "none", if you want to execute entire GANonymizerV2')
        else:
            omask = self._exec_module(self.config.mask_mode, 'omask',
                    self.mc.entire_mask, img, semseg_map)

        # visualize the mask overlayed image
        omask3c = np.stack([omask, np.zeros_like(omask), np.zeros_like(omask)], axis=-1)
        overlay = (img * 0.7 + omask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Object Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_omask_overlayed.' + self.fext)

        return omask

    def _split_object(self, omask):
        # split object mask
        print('===== Object Split =====')
        if self.config.split_mode is 'none':
            fmask, labelmap = omask, omask
        else:
            fmask, labelmap  = self._exec_module(self.config.split_mode,
                                                 ['fmask', 'olabelmap'],
                                                 self.op.split,
                                                 omask)
        return fmask, labelmap

    def _detect_shadow(self, img, labelmap):
        # shadow detection
        print('===== Shadow Detection =====')
        if self.config.shadow_mode is 'none':
            smask, labelmap = np.zeros_like(labelmap).astype(np.uint8), labelmap
        else:
            smask, labelmap = self._exec_module(self.config.shadow_mode,
                                                ['smask', 'slabelmap'],
                                                self.sd.detect,
                                                img, labelmap)
        # visualize the mask overlayed image
        smask3c = np.stack([smask, np.zeros_like(smask), np.zeros_like(smask)], axis=-1)
        overlay = (img * 0.7 + smask3c * 0.3).astype(np.uint8)
        self.debugger.img(overlay, 'Shadow Mask Overlayed Image')
        self.debugger.imsave(overlay, self.fname + '_smask_overlayed.' + self.fext)

        return smask, labelmap

    def _divide_mask(self, img, mask, labelmap):
        # pseudo mask division
        print('===== Pseudo Mask Division =====')
        if self.config.divide_mode is 'none':
            divimg = img
            divmask = mask
        else:
            divimg, divmask = self._exec_module(self.config.divide_mode,
                                                ['divimg', 'divmask'],
                                                self.md.divide,
                                                img, mask, labelmap, self.fname, self.fext)
        return divimg, divmask

    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting and Edge Inpainting =====')
        if self.config.inpaint_mode is 'none':
            raise RuntimeError('inpaint_mode should not be "none", if you want to execute entire GANonymizerV2')
        else:
            inpainted, inpainted_edge, edge = self._exec_module(self.config.inpaint_mode,
                                                                ['inpaint', 'inpaint_edge', 'edge'],
                                                                self.ii.inpaint, img, mask)
        return inpainted

    def _exec_module(self, mode, names, func, *args):
        if isinstance(names, str):
            path = os.path.join(self.config.checkpoint, self.fname + '_{}.'.format(names) + 'pkl')
        else:
            paths = []
            for name in names:
                paths.append(os.path.join(self.config.checkpoint,
                    self.fname + '_{}.'.format(name) + 'pkl'))

        if mode in ['exec', 'debug']:
            if isinstance(names, str):
                result = func(*args)
            else:
                results = list(func(*args))
        elif mode is 'pass':
            if isinstance(names, str):
                with open(path, mode='rb') as f:
                    result = pickle.load(f)
            else:
                results = []
                for path in paths:
                    with open(path, mode='rb') as f:
                        results.append(pickle.load(f))
        elif mode is 'save':
            if isinstance(names, str):
                result = func(*args)
                with open(path, mode='wb') as f:
                    pickle.dump(result, f)
            else:
                results = list(func(*args))
                for res, path in zip(results, paths):
                    with open(path, mode='wb') as f:
                        pickle.dump(res, f)

        # visualization
        if isinstance(names, str):
            self.debugger.img(result, names)
            self.debugger.imsave(result, self.fname + '_{}.'.format(names) + self.fext)
            return result
        else:
            for res, name in zip(results, names):
                self.debugger.img(res, name)
                self.debugger.imsave(res, self.fname + '_{}.'.format(name) + self.fext)
            return results

