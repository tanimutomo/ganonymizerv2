import cv2
import numpy as np
from scipy import ndimage
from collections import Counter
from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker

from .utils import expand_mask, detect_object, Debugger


class MaskDivider:
    def __init__(self, config, inpainter):
        self.config = config
        self.debugger = Debugger(config.divide_mode, save_dir=config.checkpoint)
        self.inpainter = inpainter
    

    def divide(self, img, mask):
        # separate object in the mask using random walker algorithm
        obj_labelmap = self._separete_objects(mask)

        # integrate separted small objects into big one
        obj_labelmap = self._remove_sml_obj(obj_labelmap)


    def _remove_sml_obj(self, obj_labelmap):
        # integrate separted small objects into big one
        max_label = np.max(obj_labelmap)

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

        self.debugger.img(obj_labelmap, 'output of object label map')


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
