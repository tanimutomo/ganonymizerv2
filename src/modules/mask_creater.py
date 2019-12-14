import cv2
import numpy as np
import torch

from .utils import Debugger, labels, detect_object

class MaskCreater:
    def __init__(self, config):
        self.config = config
        self.thresh = 3
        self.debugger = Debugger(config.mask_mode, save_dir=config.checkpoint)

        self.target_ids = list()
        for label in labels:
            if ((label.category is 'vehicle' or label.category is 'human')
                    and label.trainId is not 19):
                self.target_ids.append(label.trainId)

    def entire_mask(self, segmap):
        obj_mask = segmap.clone()
        for id_ in self.target_ids:
            obj_mask = torch.where(obj_mask==id_
                                   torch.full_like(segmap, 255),
                                   obj_mask,
                                   dtype=torch.uint8)
        return  torch.where(obj_mask==255,
                            torch.full_like(segmap, 255),
                            torch.full_like(segmap, 0)
                            dtype=torch.uint8)

    def separated_mask(self, img, segmap, crop_rate):
        mask, _ = self.mask(img, segmap, labels)
        H, W = mask.shape
        label_map, stats, labels = detect_object(mask)
        inputs = []
        
        for label, stat in zip(labels, stats):
            box = np.array([0, 0, 0, 0])
            obj = np.where(label_map==label, 255, 0).astype(np.uint8)
            tl = np.array([stat[0], stat[1]])
            w, h = stat[2], stat[3]
            br = np.array([stat[0] + w, stat[1] + h])
            center = np.floor(((br + tl) / 2)).astype(np.int32)
            area = stat[4]

            if w*3 > 256:
                box[0], box[2] = self._crop_obj(tl[0], br[0],
                        int(w / crop_rate), 0, W)

            else:
                box[0], box[2] = self._crop_obj(center[0], center[0], 128, 0, W)
                    
            if h*3 > 256:
                box[1], box[3] = self._crop_obj(tl[1], br[1],
                        int(h / crop_rate), 0, H)

            else:
                box[1], box[3] = self._crop_obj(center[1], center[1], 128, 0, H)

            img_part = img[box[1]:box[3], box[0]:box[2], :]
            mask_part = obj[box[1]:box[3], box[0]:box[2]]

            img_part = self._adjust_imsize(img_part)
            mask_part = self._adjust_imsize(mask_part)

            assert (img_part.shape[0], img_part.shape[1]) == (mask_part.shape[0], mask_part.shape[1])
            inputs.append({'img': img_part, 'mask': mask_part, 
                'box': box, 'new_size': mask_part.shape, 'area': area})

        return inputs

    def _crop_obj(self, top, bot, base, min_thresh, max_thresh):
        if top - base >= min_thresh:
            ntop = top - base
            extra = 0
        else:
            ntop = min_thresh
            extra = min_thresh - (top - base)

        if bot + base + extra < max_thresh:
            nbot = bot + base + extra
            extra = 0
        else:
            nbot = max_thresh
            extra = bot + base + extra - max_thresh

        ntop = ntop - extra if ntop - extra >= min_thresh else min_thresh

        return ntop, nbot

    def _adjust_imsize(self, img):
        size = np.array([img.shape[0], img.shape[1]])
        nsize = np.array([0, 0])
        argmax = np.argmax(size)
        if size[argmax] > 300:
            argmin = np.argmin(size)
            rate = 300 / size[argmax]
            nsize[argmax] = 300
            nsize[argmin] = np.ceil(size[argmin] * rate).astype(np.int32)
        else:
            nsize = size.copy()

        nsize[0] = nsize[0] if nsize[0] % 4 == 0 else nsize[0] - (nsize[0] % 4)
        nsize[1] = nsize[1] if nsize[1] % 4 == 0 else nsize[1] - (nsize[1] % 4)
        out = cv2.resize(img, (nsize[1], nsize[0]))

        return out

