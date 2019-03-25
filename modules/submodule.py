import cv2
import skimage
import numpy as np
from scipy import ndimage

from skimage.morphology import label
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker, mark_boundaries


def tensor_img_to_numpy(tensor):
    array = tensor.numpy()
    array = np.transpose(array, (1, 2, 0))
    return array


def detect_object(img):
    num, label_map, stats, _ = cv2.connectedComponentsWithStats(img)
    # stat is [tl_x, tl_y, w, h, area]
    label_list = [i+1 for i in range(num - 1)]
    return label_map, stats[1:], label_list


def expand_mask(mask, width):
    if width == 0: return mask
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, 255, width) 
    return mask.astype(np.uint8)


def write_labels(img, segmap, size):
    out = mark_boundaries(img, segmap)
    labels = np.unique(segmap)
    for label in labels:
        ys, xs = np.where(segmap==label)
        my, mx = np.median(ys).astype(np.int32), np.median(xs).astype(np.int32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        out = cv2.putText(out, str(label), (mx - 10, my), font, size, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def separate_objects(mask, debugger):
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
    debugger.img(labelmap, 'labelmap')

    # restore missing objects
    labelmap = _restore_missings(labelmap, mask, debugger)
    # integrate separted small objects into big one
    labelmap = _remove_sml_obj(labelmap, debugger)

    return labelmap

def _restore_missings(labelmap, mask, debugger):
    lmax = np.max(labelmap)
    debugger.param(lmax, 'max value of label')
    debugger.param(labelmap, 'labelmap')
    labeled = np.where(labelmap > 0, 1, 0).astype(np.int32)
    mask = np.where(mask > 0, 1, 0).astype(np.int32)
    unlabeled = mask - labeled
    debugger.img(unlabeled, 'unlabeled')
    unlabeled_map, _, labels = detect_object(unlabeled.astype(np.uint8))
    for label in labels:
        debugger.param(label, 'label number')
        obj = np.where(unlabeled_map == label, lmax + label, 0).astype(labelmap.dtype)
        debugger.param(label, 'old label')
        debugger.param(np.max(obj), 'new label')
        debugger.img(obj, 'object')
        labelmap += obj

    debugger.img(labelmap, 'restored labelmap')
    debugger.img(write_labels(labelmap, labelmap, 1),
            'restored labelmap with label number')
    return labelmap

def _remove_sml_obj( labelmap, debugger):
    # integrate separted small objects into big one

    removed_labels = []
    # start from object label 1 (0 is not object)
    for label in range(1, np.max(labelmap) + 1):
        debugger.param(label, 'label number')
        # create each object mask
        objmap = np.where(labelmap == label, 1, 0).astype(np.uint8)
        objmap_wb = np.where(objmap == 1, 0, 1).astype(np.uint8)
        # debugger.img(objmap, 'object map')
        debugger.img(objmap_wb, 'object map (white blob)')
        
        # calcurate area of the object
        area = np.sum(objmap)

        # object size filter if an object is bigger than threshl, the following process is passed
        if area > labelmap.shape[0] * labelmap.shape[1] * config.obj_sml_thresh: continue

        # get contours(x, y) of the object
        try:
            contours, _ = cv2.findContours(objmap_wb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            debugger.matrix(contours[1], 'contours')
        except:
            continue

        contours = contours[1].squeeze().transpose()

        # if object is located in the image edge, remove image edge point from contours
        if contours.shape[1] >= labelmap.shape[0] * 2 + labelmap.shape[1] * 2:
            debugger.matrix(contours, 'before contours')
            for value in [0, labelmap.shape[0] - 1, labelmap.shape[1] - 1]:
                contours = np.where(contours == value, -1, contours)
            debugger.matrix(np.where(contours[0] == -1)[0], '0 delete index')
            debugger.matrix(np.where(contours[1] == -1)[0], '0 delete index')
            delidx = np.unique(np.concatenate([np.where(contours[0] == -1)[0],
                np.where(contours[1] == -1)[0]]))
            debugger.matrix(delidx, 'delete index')
            contours = np.delete(contours, delidx, axis=-1)
            debugger.matrix(contours, 'after contours')

        # detect surrounding object label
        debugger.matrix(contours, 'aranged contours')
        sur_labels = labelmap[contours[1], contours[0]]
        debugger.matrix(sur_labels, 'surrounding labels')

        # delete 0 label (background) from surrounding labels
        sur_labels = np.delete(sur_labels, np.where(sur_labels == 0)[0])

        # if all surrounding labels is 0, the following process is passed
        if sur_labels.size == 0: continue

        # calcurate most labels in surroundings
        debugger.matrix(sur_labels, 'deleted 0 sur_labels')
        label_count = Counter(sur_labels.tolist())
        most_sur_label = max(label_count, key=label_count.get)
        debugger.param(most_sur_label, 'most surrounding label')

        # integrate this label object into most surrounding label
        labelmap = np.where(labelmap == label, most_sur_label, labelmap)
        removed_labels.append(label)

    # make labels order sequential
    for label in removed_labels:
        labelmap = np.where(labelmap > label, labelmap - 1, labelmap)

    debugger.img(labelmap, 'output of object label map')
    debugger.img(write_labels(labelmap, labelmap, 1),
            'removed small object labelmap')
    return labelmap
