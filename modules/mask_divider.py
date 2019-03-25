import os
import cv2
import numpy as np
from PIL import Image

from .utils import Debugger, detect_object, write_labels


class MaskDivider:
    def __init__(self, config, inpainter):
        self.config = config
        self.debugger = Debugger(config.divide_mode, save_dir=config.checkpoint)
        self.inpainter = inpainter

    def divide(self, img, mask, labelmap, fname, fext):
        # get large object
        objects = self._classify_object(labelmap)

        # apply inpainting to resized image for each large object
        objects = self._get_sml_inpainted(objects, img, mask, fname, fext)

        # calcurate the lattice position and create new inputs
        new_img, new_mask = self._divide_inputs(objects, img, mask)

        return new_img, new_mask


    def _classify_object(self, labelmap):
        objects = []
        for label in range(1, np.max(labelmap) + 1):
            self.debugger.param(label, 'label number')
            # get object mask
            objmask = np.where(labelmap == label, 1, 0)
            self.debugger.img(objmask, 'objmask')

            # calcurate object area
            area = np.sum(objmask)
            self.debugger.param(area, 'area')

            # filtering by area
            if area < labelmap.shape[0] * labelmap.shape[1] * self.config.obj_sml_thresh:
                objects.append({'large': False, 'mask': objmask})
                continue

            # calcurate object box and boxarea
            ys, xs = np.where(objmask == 1)
            box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            self.debugger.param(box, 'box')
            height, width = box[3] - box[1], box[2] - box[0]
            self.debugger.param(width, 'width')
            self.debugger.param(height, 'height')

            # filtering by box width, height
            if width < self.config.obj_wh_thresh or height < self.config.obj_wh_thresh:
                objects.append({'large': False, 'mask': objmask})
                continue

            # calcurate the density
            density = area / (width * height)
            self.debugger.param(density, 'density')

            # filtering by density
            if density < self.config.obj_density_thresh:
                objects.append({'large': False, 'mask': objmask})
            else:
                objects.append({
                        'large': True,
                        'mask': objmask,
                        'box': box,
                        'width': width,
                        'height': height
                        })

        self.debugger.param(objects, 'objects')
        return objects

    def _get_sml_inpainted(self, objects, img, mask, fname, fext):
        H, W = img.shape[0], img.shape[1]

        # apply inpaint each resized image
        for idx, obj in enumerate(objects):
            # don't apply to small object
            if not obj['large']: continue

            # copy original image and mask for resizing 
            resimg, resmask = img.copy(), mask.copy()

            # calcurate new height and width
            height, width = obj['height'], obj['width']
            base = max(width, height)
            ratio = self.config.obj_wh_thresh / base
            new_h, new_w = int(H * ratio), int(W * ratio)
            new_h = new_h if new_h % 4 == 0 else new_h - (new_h % 4)
            new_w = new_w if new_w % 4 == 0 else new_w - (new_w % 4)
            self.debugger.param(ratio, 'ratio')
            self.debugger.param(new_w, 'new w')
            self.debugger.param(new_h, 'new h')

            # resize inputs
            resimg = cv2.resize(resimg, (new_w, new_h)).astype(np.uint8)
            resmask = cv2.resize(resmask, (new_w, new_h)).astype(np.uint8)
            resmask = np.where(resmask > 127, 255, 0)
            self.debugger.img(resimg, 'resized image')
            self.debugger.img(resmask, 'resized mask')

            # apply inpainting
            paths = [
                    os.path.join(self.config.checkpoint, fname + '_{}_sml_inp.'.format(idx) + fext),
                    os.path.join(self.config.checkpoint, fname + '_{}_sml_inpe.'.format(idx) + fext),
                    os.path.join(self.config.checkpoint, fname + '_{}_sml_edge.'.format(idx) + fext)
                    ]
            self.debugger.param(paths, 'paths')
            if os.path.exists(paths[0]) and self.debugger.mode != 'debug':
                inpainted = np.array(Image.open(paths[0]))
                inpainted_edge = np.array(Image.open(paths[1]))
                edge = np.array(Image.open(paths[2]))
            else:
                inpainted, inpainted_edge, edge = self.inpainter.inpaint(resimg, resmask) 
                Image.fromarray(inpainted).save(paths[0])
                Image.fromarray(inpainted_edge).save(paths[1])
                Image.fromarray(edge).save(paths[2])

            self.debugger.img(inpainted, 'inpainted')
            self.debugger.img(inpainted_edge, 'inpainted_edge')
            self.debugger.img(edge, 'edge')

            # resize output to original size
            inpainted = cv2.resize(inpainted, (W, H))
            inpainted_edge = cv2.resize(inpainted_edge, (W, H))
            edge = cv2.resize(edge, (W, H))
            self.debugger.img(inpainted, 'inpainted')
            self.debugger.img(inpainted_edge, 'inpainted_edge')
            self.debugger.img(edge, 'edge')

            # add output to objects dict
            obj['inpainted'] = inpainted
            obj['inpainted_edge'] = inpainted_edge
            obj['edge'] = edge

        self.debugger.param(objects, 'large objects')
        return objects

    def _divide_inputs(self, objects, img, mask):
        for idx, obj in enumerate(objects):
            # don't apply to small object
            if obj['large']:
                self.debugger.img(obj['mask'], 'check object')
                x, y, _, _ = obj['box']
                h, w = obj['height'], obj['width']
                self.debugger.param(h, 'height')
                self.debugger.param(w, 'width')

                # create lattice mask image
                lattice = np.zeros_like(obj['mask']).astype(np.uint8)
                vline_w, hline_w = int(w / 16), int(h / 16)
                self.debugger.param(vline_w, 'vline width')
                self.debugger.param(hline_w, 'hline width')
                # draw vertical line
                lattice = cv2.line(lattice,
                        (x + int(w * (5/16)), y + int(h * (1/8))),
                        (x + int(w * (5/16)), y + int(h * (7/8))),
                        1, vline_w)
                lattice = cv2.line(lattice,
                        (x + int(w * (11/16)), y + int(h * (1/8))),
                        (x + int(w * (11/16)), y + int(h * (7/8))),
                        1, vline_w)
                # draw horizontal line
                lattice = cv2.line(lattice,
                        (x + int(w * (1/8)), y + int(h * (5/16))),
                        (x + int(w * (7/8)), y + int(h * (5/16))),
                        1, hline_w)
                lattice = cv2.line(lattice,
                        (x + int(w * (1/8)), y + int(h * (11/16))),
                        (x + int(w * (7/8)), y + int(h * (11/16))),
                        1, hline_w)

                lattice = np.where(obj['mask'] == 1, lattice, 0)
                self.debugger.img(lattice, 'lattice image')
                lattice3c = np.stack([lattice for _ in range(3)], axis=-1).astype(np.uint8)
                # insert inpainted image to lattice part in black image
                imglat = np.where(lattice3c == 1, obj['inpainted'], lattice3c)
                # insert this object mask image to not lattice part in black image
                masklat = np.where(lattice == 1, 0, obj['mask'])

            else:
                imglat = np.zeros_like(img)
                masklat = obj['mask']

            self.debugger.img(imglat, 'lattice image')
            self.debugger.img(masklat, 'lattice mask')

            # unite all object lattice part image
            if idx == 0:
                new_img = imglat
                new_mask = masklat
            else:
                new_img += imglat
                new_mask += masklat

            self.debugger.img(new_img, 'new_img')
            self.debugger.img(new_mask, 'new_mask')

        # substitute original image for not lattice part
        new_img = np.where(new_img == 0, img, new_img)
        new_mask = np.where(new_mask > 0, 255, 0)

        self.debugger.img(new_img, 'new_img')
        self.debugger.img(new_mask, 'new_mask')

        return new_img.astype(np.uint8), new_mask.astype(np.uint8)


