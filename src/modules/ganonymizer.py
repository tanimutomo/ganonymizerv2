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
# from .object_spliter import ObjectSpliter
# from .shadow_detecter import ShadowDetecter
# from .mask_divider import MaskDivider
from .inpainter import ImageInpainter
# from .randmask_creater import RandMaskCreater
from .utils import Debugger, label_img_to_color, expand_mask


class GANonymizer:
    def __init__(self, config, device):
        self.config = config
        self.devide = device
        self.debugger = Debugger(config.main_mode, save_dir=config.checkpoint)

        self.ss = SemanticSegmenter(config, device)
        self.mc = MaskCreater(config)
        # self.op = ObjectSpliter(config)
        # self.sd = ShadowDetecter(config)
        self.ii = ImageInpainter(config, device)
        # self.md = MaskDivider(config, self.ii)
        # self.rm = RandMaskCreater(config)

        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def reload_config(self, config):
        self.config = config
        self.debugger = Debugger(config.main_mode, save_dir=config.checkpoint)
        self.ss = SemanticSegmenter(config, device)
        self.mc = MaskCreater(config)
        # self.op = ObjectSpliter(config)
        # self.sd = ShadowDetecter(config)
        self.ii = ImageInpainter(config, device)
        # self.md = MaskDivider(config, self.ii)
        # self.rm = RandMaskCreater(config)
    
    def predict(self, pil_img):
        img = self.to_tensor(pil_img) # to torch.tensor in [0, 1]
        self.debugger.img(img, 'Input Image')

        # semantic segmentation for detecting dynamic objects and creating mask
        label_map = self._semseg(img)

        # get object mask
        mask = self._object_mask(img, label_map)

        # # get hole filled object mask and object labelmap
        # omask, labelmap = self._split_object(omask)

        # # detect shadow area and add shadow area to object mask
        # smask, labelmap = self._detect_shadow(img, labelmap)
        # mask, labelmap = self._combine_masks(img, omask, smask, labelmap)

        # # create random mask for evaluate the effect of PMD
        # mask, labelmap = self._random_mask(mask, labelmap)

        # # Psuedo Mask Division
        # divimg, divmask = self._divide_mask(img, mask, labelmap)

        # resize inputs
        img, mask = self._resize(img, mask)

        # image and edge inpainting
        out = self._inpaint(img, mask)

        # save output image
        self._save_output(out)

        return out

    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')
        if self.config.semseg_mode is 'none':
            raise RuntimeError('semseg_mode should not be "none", if you want to execute entire GANonymizerV2')
        else:
            semseg_map = self._exec_module(
                self.config.semseg_mode,
                'segmap',
                self.ss.predict,
                img
            )

        vis, lc_img = label_img_to_color(semseg_map)
        self.debugger.img(vis, 'color semseg map')
        self.debugger.img(lc_img, 'label color map')

        return semseg_map

    def _object_mask(self, img, label_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        if self.config.mask_mode is 'none':
            raise RuntimeError('mask_mode should not be "none", if you want to execute entire GANonymizerV2')
        else:
            omask = self._exec_module(self.config.mask_mode, 'omask',
                                      self.mc.create_mask, label_map) # shape=(h, w) # dtype=torch.uint8

        # visualize the mask overlayed image
        omask3c = torch.stack([omask, torch.zeros_like(omask), torch.zeros_like(omask)], dim=0)
        img = (img * 255).to(torch.uint8)
        overlay = (img * 0.8 + omask3c * 0.2).to(torch.uint8)
        self.debugger.img(overlay, 'Object Mask Overlayed Image')
        self.debugger.imsave(overlay, self.config.fname + '_omask_overlayed.' + self.config.fext)
        return omask

    def _resize(self, img, mask):
        # resize inputs
        new_h = img.shape[1] // self.config.resize_factor
        new_w = img.shape[2] // self.config.resize_factor
        img = self.to_pil(img).resize((new_w, new_h))
        mask = self.to_pil(mask).resize((new_w, new_h))
        return self.to_tensor(img), self.to_tensor(mask)

    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting =====')
        if self.config.inpaint_mode is 'none':
            raise RuntimeError('inpaint_mode should NOT be "none", if you want to execute entire GANonymizerV2')
        else:
            inpainted, inpainted_edge, edge = self._exec_module(
                self.config.inpaint_mode,
                ['inpaint', 'inpaint_edge', 'edge'],
                self.ii.inpaint,
                img, mask
            )
        return inpainted

    # def _combine_masks(self, img, omask, smask, labelmap):
    #     # check omask and smak shape
    #     assert omask.shape == smask.shape

    #     # combine the object mask and the shadow mask
    #     mask = np.where(omask + smask > 0, 255, 0).astype(np.uint8)
    #     mask = expand_mask(mask, self.config.expand_width)
    #     self.debugger.img(mask, 'Mask (Object Mask + Shadow Mask)')
    #     self.debugger.imsave(mask, self.config.fname + '_mask.' + self.config.fext)

    #     # expand labelmap
    #     for label in range(1, np.max(labelmap)+1):
    #         objmap = np.where(labelmap == label, 1, 0)
    #         objmap = expand_mask(objmap, self.config.expand_width)
    #         objmap = (objmap / 255 * label).astype(np.int32)
    #         labelmap = np.where(labelmap == 0, objmap, labelmap)
    #     self.debugger.img(labelmap, 'Expanded LabelMap')

    #     # visualization the mask overlayed image
    #     mask3c = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
    #     overlay = (img * 0.7 + mask3c * 0.3).astype(np.uint8)
    #     self.debugger.img(overlay, 'Mask Overlayed Image')
    #     self.debugger.imsave(overlay, self.config.fname + '_mask_overlayed.' + self.config.fext)

    #     # image with mask
    #     mask3c = np.stack([mask for _ in range(3)], axis=-1)
    #     img_with_mask = np.where(mask3c==255, mask3c, img).astype(np.uint8)
    #     self.debugger.img(img_with_mask, 'Image with Mask')
    #     self.debugger.imsave(img_with_mask, self.config.fname + '_img_with_mask.' + self.config.fext)

    #     return mask, labelmap

    # def _split_object(self, omask):
    #     # split object mask
    #     print('===== Object Split =====')
    #     if self.config.split_mode is 'none':
    #         fmask, labelmap = omask, omask
    #     else:
    #         fmask, labelmap  = self._exec_module(self.config.split_mode,
    #                                              ['fmask', 'olabelmap'],
    #                                              self.op.split,
    #                                              omask)
    #     return fmask, labelmap

    # def _detect_shadow(self, img, labelmap):
    #     # shadow detection
    #     print('===== Shadow Detection =====')
    #     if self.config.shadow_mode is 'none':
    #         smask, labelmap = np.zeros_like(labelmap).astype(np.uint8), labelmap
    #     else:
    #         smask, labelmap = self._exec_module(self.config.shadow_mode,
    #                                             ['smask', 'slabelmap'],
    #                                             self.sd.detect,
    #                                             img, labelmap)
    #     # visualize the mask overlayed image
    #     smask3c = np.stack([smask, np.zeros_like(smask), np.zeros_like(smask)], axis=-1)
    #     overlay = (img * 0.7 + smask3c * 0.3).astype(np.uint8)
    #     self.debugger.img(overlay, 'Shadow Mask Overlayed Image')
    #     self.debugger.imsave(overlay, self.config.fname + '_smask_overlayed.' + self.config.fext)

    #     return smask, labelmap

    # def _divide_mask(self, img, mask, labelmap):
    #     # pseudo mask division
    #     print('===== Pseudo Mask Division =====')
    #     if self.config.divide_mode is 'none':
    #         divimg = img.copy()
    #         divmask = mask.copy()
    #     else:
    #         divimg, divmask = self._exec_module(self.config.divide_mode,
    #                                             ['divimg', 'divmask'],
    #                                             self.md.divide,
    #                                             img, mask, labelmap, self.config.fname, self.config.fext)
    #     return divimg, divmask

    # def _random_mask(self, mask, labelmap):
    #     # random mask for evaluating pmd effectness
    #     print('===== Create Random Mask =====')
    #     if self.config.random_mode is 'none':
    #         rmask = mask
    #         labelmap = labelmap
    #     else:
    #         rmask, labelmap = self._exec_module(self.config.random_mode,
    #                                             'rmask',
    #                                             self.rm.sample,
    #                                             mask)
    #     return rmask, labelmap

    def _exec_module(self, mode, names, func, *args):
        if isinstance(names, str):
            path = os.path.join(self.config.checkpoint, self.config.fname + '_{}.'.format(names) + 'pkl')
        else:
            paths = []
            for name in names:
                paths.append(os.path.join(self.config.checkpoint,
                    self.config.fname + '_{}.'.format(name) + 'pkl'))

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
            self.debugger.imsave(result, self.config.fname + '_{}.'.format(names) + self.config.fext)
            return result
        else:
            for res, name in zip(results, names):
                self.debugger.img(res, name)
                self.debugger.imsave(res, self.config.fname + '_{}.'.format(name) + self.config.fext)
            return results

    def _save_output(self, out):
        print(out.shape)
        outimg = self.to_pil(out)
        shadow = 'off' if self.config.shadow_mode is 'none' else 'on'
        pmd = 'off' if self.config.divide_mode is 'none' else 'on'
        
        if self.config.mode == 'pmd':
            savepath = os.path.join(self.config.eval_pmd_path,
                                    '{}_out_{}.{}'.format(self.config.fname, pmd, self.config.fext))
        elif self.config.mode == 'demo':
            savepath = os.path.join(self.config.output,
                                    '{}_out.{}'.format(self.config.fname, self.config.fext))
        else:
            savepath = os.path.join(self.config.output,
                                    '{}_out_expanded_{}_shadow_{}_pmd_{}.{}'.format(
                                        self.config.fname, self.config.expand_width,
                                        shadow, pmd, self.config.fext))
        if self.config.mode != "video":
            outimg.save(savepath)
