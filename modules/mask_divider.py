import cv2
import numpy as np
from scipy import ndimage
from collections import Counter
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker, relabel_sequential

from .utils import expand_mask, detect_object, Debugger


class MaskDivider:
    def __init__(self, config, inpainter):
        self.config = config
        self.debugger = Debugger('exec', save_dir=config.checkpoint)
        self.debugger2 = Debugger('exec', save_dir=config.checkpoint)
        self.debugger3 = Debugger(config.divide_mode, save_dir=config.checkpoint)
        self.inpainter = inpainter
    

    def divide(self, img, mask):
        # separate object in the mask using random walker algorithm
        obj_labelmap = self._separate_objects(mask)

        # integrate separted small objects into big one
        obj_labelmap = self._remove_sml_obj(obj_labelmap)

        # get large object
        large_objects = self._large_object(obj_labelmap)

        # apply inpainting to resized image for each large object
        large_objects = self._get_sml_inpainted(large_objects, img, mask)


    def _separate_objects(self, mask):
        # separate object in the mask using random walker algorithm
        # In obj_labelmap, -1 is background, 0 is none, object is from 1
        mask = np.where(mask > 0, 1, 0).astype(np.bool)

        ksize = int((mask.shape[0] + mask.shape[1]) / 30)
        distance = ndimage.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, indices=False, 
                footprint=np.ones((ksize, ksize)), labels=mask)
        markers = label(local_maxi)
        markers[~mask] = -1
        labelmap = random_walker(mask, markers)
        
        self.debugger.img(labelmap, 'labelmap', gray=True)

        return labelmap


    def _remove_sml_obj(self, obj_labelmap):
        # integrate separted small objects into big one
        max_label = np.max(obj_labelmap)

        removed_labels = []
        # start from object label 1 (-1 and 0 is not object)
        for label in range(1, max_label):
            # create each object mask
            objmap = np.where(obj_labelmap == label, 1, 0).astype(np.uint8)
            objmap_wb = np.where(objmap == 1, 0, 1).astype(np.uint8)
            self.debugger.img(objmap, 'object map')
            self.debugger.img(objmap_wb, 'object map (white blob)')
            
            # calcurate area of the object
            area = np.sum(objmap)

            # object size filter if an object is bigger than threshl, the following process is passed
            if area > obj_labelmap.shape[0] * obj_labelmap.shape[1] * self.config.obj_sml_thresh: continue

            # get contours(x, y) of the object
            contours, _ = cv2.findContours(objmap_wb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            self.debugger.matrix(contours[1], 'contours')

            # detect surrounding object label
            contours = contours[1].squeeze().transpose()
            self.debugger.matrix(contours, 'aranged contours')
            sur_labels = obj_labelmap[contours[1], contours[0]]
            self.debugger.matrix(sur_labels, 'surrounding labels')

            # delete -1 label (background) from surrounding labels
            sur_labels = np.delete(sur_labels, np.where(sur_labels == -1)[0])

            # if all surrounding labels is -1, the following process is passed
            if sur_labels.size == 0: continue

            # calcurate most labels in surroundings
            self.debugger.matrix(sur_labels, 'deleted -1 sur_labels')
            label_count = Counter(sur_labels.tolist())
            most_sur_label = max(label_count, key=label_count.get)
            self.debugger.param(most_sur_label, 'most surrounding label')

            # integrate this label object into most surrounding label
            obj_labelmap = np.where(obj_labelmap == label, most_sur_label, obj_labelmap)
            removed_labels.append(label)

        # make label order sequential
        for label in removed_labels:
            obj_labelmap = np.where(obj_labelmap > label, obj_labelmap - 1, obj_labelmap)

        self.debugger.img(obj_labelmap, 'output of object label map')
        return obj_labelmap


    def _large_object(self, labelmap):
        # change background label from -1 to 0
        labelmap = np.where(labelmap < 0, 0, labelmap)
        larges = []
        for label in range(1, np.max(labelmap)):
            # get object mask
            objmask = np.where(labelmap == label, 1, 0)
            self.debugger2.img(objmask, 'objmask', gray=True)

            # calcurate object area
            area = np.sum(objmask)
            self.debugger2.param(area, 'area')

            # filtering by area
            if area < labelmap.shape[0] * labelmap.shape[1] * self.config.obj_sml_thresh: continue

            # calcurate object box and boxarea
            ys, xs = np.where(objmask == 1)
            box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            self.debugger2.param(box, 'box')
            width, height = box[3] - box[1], box[2] - box[0]
            self.debugger2.param(width, 'width')
            self.debugger2.param(height, 'height')

            # filtering by box width, height
            if width < self.config.obj_wh_thresh or height < self.config.obj_wh_thresh: continue

            # calcurate the density
            density = area / (width * height)
            self.debugger2.param(density, 'density')

            # filtering by density
            if density < self.config.obj_density_thresh: continue

            larges.append({
                    'mask': objmask,
                    'box': box,
                    'width': width,
                    'height': height
                    })

        self.debugger2.param(larges, 'larges')
        return larges


    def _get_sml_inpainted(self, large_objects, img, mask):
        H, W = img.shape[0], img.shape[1]

        # apply inpaint each resized image
        for obj in large_objects:
            resimg, resmask = img.copy(), mask.copy()

            # calcurate new height and width
            height, width = obj['height'], obj['width']
            base = max(width, height)
            ratio = self.config.obj_wh_thresh / base
            new_h, new_w = int(H * ratio), int(W * ratio)
            new_h = new_h if new_h % 4 == 0 else new_h - (new_h % 4)
            new_w = new_w if new_w % 4 == 0 else new_w - (new_w % 4)
            self.debugger3.param(ratio, 'ratio')
            self.debugger3.param(new_w, 'new w')
            self.debugger3.param(new_h, 'new h')

            # resize inputs
            resimg = cv2.resize(resimg, (new_w, new_h)).astype(np.uint8)
            resmask = cv2.resize(resmask, (new_w, new_h)).astype(np.uint8)
            resmask = np.where(resmask > 127, 255, 0)
            self.debugger3.img(resimg, 'resized image')
            self.debugger3.img(resmask, 'resized mask')

            # apply inpainting
            inpainted, inpainted_edge, edge = self.inpainter.inpaint(resimg, resmask) 
            self.debugger3.img(inpainted, 'inpainted')
            self.debugger3.img(inpainted_edge, 'inpainted_edge')
            self.debugger3.img(edge, 'edge')

            # resize output to original size
            inpainted = cv2.resize(inpainted, (W, H))
            inpainted_edge = cv2.resize(inpainted_edge, (W, H))
            edge = cv2.resize(edge, (W, H))
            self.debugger3.img(inpainted, 'inpainted')
            self.debugger3.img(inpainted_edge, 'inpainted_edge')
            self.debugger3.img(edge, 'edge')

            # add output to objects dict
            obj['inpainted'] = inpainted
            obj['inpainted_edge'] = inpainted_edge
            obj['edge'] = edge

        self.debugger3.param(large_objects, 'large objects')
        return large_objects



