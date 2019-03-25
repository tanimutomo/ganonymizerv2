import cv2
import skimage
import numpy as np
from scipy import ndimage
from collections import Counter

from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker, mark_boundaries

from .utils import Debugger, detect_object, write_labels


class ObjectSpliter:
    def __init__(self, config):
        self.config = config
        self.debugger = Debugger(config.split_mode, save_dir=config.checkpoint)

    def split(self, mask):
        # separate object in the mask using random walker algorithm
        # In labelmap, -1 is background, 0 is none, object is from 1
        mask = np.where(mask > 0, 1, 0).astype(np.bool)

        ksize = int((mask.shape[0] + mask.shape[1]) / 30)
        distance = ndimage.distance_transform_edt(mask)
        local_maxi = peak_local_max(distance, indices=False, 
                footprint=np.ones((ksize, ksize)), labels=mask)
        markers = label(local_maxi)
        markers[~mask] = -1
        labelmap = random_walker(mask, markers)
        
        labelmap = np.where(labelmap < 0, 0, labelmap)

        # restore missing objects
        labelmap = self._restore_missings(labelmap, mask)
        # integrate separted small objects into big one
        labelmap = self._remove_sml_obj(labelmap)
        # fill the holes in objects
        labelmap = self._fill_hole(labelmap)
        # relabel from 1
        labelmap = self._relabel(labelmap)

        self.debugger.img(labelmap, 'labelmap')
        self.debugger.param(np.unique(labelmap), 'all labels in labelmap')
        return np.where(labelmap > 0, 255, 0).astype(np.uint8), labelmap

    def _restore_missings(self, labelmap, mask):
        lmax = np.max(labelmap)
        self.debugger.param(lmax, 'max value of label')
        self.debugger.param(labelmap, 'labelmap')
        labeled = np.where(labelmap > 0, 1, 0).astype(np.int32)
        mask = np.where(mask > 0, 1, 0).astype(np.int32)
        unlabeled = mask - labeled
        self.debugger.img(unlabeled, 'unlabeled')
        unlabeled_map, _, labels = detect_object(unlabeled.astype(np.uint8), self.debugger)
        for label in labels:
            self.debugger.param(label, 'label number')
            obj = np.where(unlabeled_map == label, lmax + label, 0).astype(labelmap.dtype)
            self.debugger.param(label, 'old label')
            self.debugger.param(np.max(obj), 'new label')
            self.debugger.img(obj, 'object')
            labelmap += obj

        self.debugger.img(labelmap, 'restored labelmap')
        self.debugger.img(write_labels(labelmap, labelmap, 1),
                'restored labelmap with label number')
        return labelmap

    def _remove_sml_obj(self, labelmap):
        # integrate separted small objects into big one

        removed_labels = []
        # start from object label 1 (0 is not object)
        for label in range(1, np.max(labelmap) + 1):
            self.debugger.param(label, 'label number')
            # create each object mask
            objmap = np.where(labelmap == label, 1, 0).astype(np.uint8)
            objmap_wb = np.where(objmap == 1, 0, 1).astype(np.uint8)
            self.debugger.img(objmap_wb, 'object map (white blob)')
            if np.all(objmap_wb == 1):
                continue
            
            # calcurate area of the object
            area = np.sum(objmap)

            # get contours(x, y) of the object
            try:
                all_contours, _ = cv2.findContours(objmap_wb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            except:
                continue

            for contour in all_contours:
                self.debugger.matrix(contour, 'contour')

            # the case that object is Separated to the edge of the image
            if all_contours[0].shape[0] == labelmap.shape[0]*2 + labelmap.shape[1]*2 - 4:
                contours = all_contours[1].squeeze().transpose()
                inside = all_contours[2:]
            # the case that object is Adjust to the edge of the image
            else:
                maxidx, maxlen = 0, 0
                for idx, cnt in enumerate(all_contours):
                    if cnt.shape[0] > maxlen:
                        maxidx, maxlen = idx, cnt.shape[0]

                contours = all_contours.pop(maxidx).squeeze().transpose()
                inside = all_contours

                # if object is located in the image edge, remove image edge point from contours
                self.debugger.matrix(contours, 'before contours')
                for value in [0, labelmap.shape[0] - 1, labelmap.shape[1] - 1]:
                    contours = np.where(contours == value, -1, contours)
                self.debugger.matrix(np.where(contours[0] == -1)[0], '0 delete index')
                self.debugger.matrix(np.where(contours[1] == -1)[0], '0 delete index')
                delidx = np.unique(np.concatenate([np.where(contours[0] == -1)[0],
                    np.where(contours[1] == -1)[0]]))
                self.debugger.matrix(delidx, 'delete index')
                contours = np.delete(contours, delidx, axis=-1)
                self.debugger.matrix(contours, 'after contours')

            # # fill in the holes inside object
            # labelmap = cv2.drawContours(labelmap, inside, -1, label, -1)

            # detect surrounding object label
            self.debugger.matrix(contours, 'aranged contours')
            sur_labels = labelmap[contours[1], contours[0]]
            self.debugger.matrix(sur_labels, 'surrounding labels')

            bgidx = np.where(sur_labels == 0)[0]
            self.debugger.param(len(bgidx), 'background index length')
            self.debugger.param(bgidx.size / sur_labels.size, 'background opacity')
            if bgidx.size / sur_labels.size > 1 - self.config.obj_sep_thresh:
                continue

            # delete 0 label (background) from surrounding labels
            sur_labels = np.delete(sur_labels, bgidx)
            # if all surrounding labels is 0, the following process is passed
            if sur_labels.size == 0:
                continue

            # calcurate most labels in surroundings
            self.debugger.matrix(sur_labels, 'deleted 0 sur_labels')
            label_count = Counter(sur_labels.tolist())
            most_sur_label = max(label_count, key=label_count.get)
            self.debugger.param(most_sur_label, 'most surrounding label')

            # integrate this label object into most surrounding label
            labelmap = np.where(labelmap == label, most_sur_label, labelmap)
            removed_labels.append(label)

        self.debugger.img(labelmap, 'output of object label map')
        self.debugger.img(write_labels(labelmap, labelmap, 1),
                'removed small object labelmap')

        return labelmap

    def _fill_hole(self, labelmap):
        for label in range(1, np.max(labelmap)):
            objmap = np.where(labelmap == label, 0, 1).astype(np.uint8)
            if np.all(objmap == 1):
                continue
            all_contours, _ = cv2.findContours(objmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # the case that object is Separated to the edge of the image
            if all_contours[0].shape[0] == labelmap.shape[0]*2 + labelmap.shape[1]*2 - 4:
                inside = all_contours[2:]
            # the case that object is Adjust to the edge of the image
            else:
                maxidx, maxlen = 0, 0
                for idx, cnt in enumerate(all_contours):
                    if cnt.shape[0] > maxlen:
                        maxidx, maxlen = idx, cnt.shape[0]

                all_contours.pop(maxidx).squeeze().transpose()
                inside = all_contours

            # fill in the holes inside object
            labelmap = cv2.drawContours(labelmap, inside, -1, label, -1)

        return labelmap

    def _relabel(self, labelmap):
        for new_label, label in enumerate(np.unique(labelmap)):
            labelmap = np.where(labelmap == label, new_label, labelmap)
        return labelmap
    
