import cv2
import numpy as np
import torch
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries

class ShadowDetecter:
    def __init__(self, debug):
        self.debug = debug

    
    def detect(self, img, segmap, labels):
        img_edges = cv2.Canny(img, 100, 200, L2gradient=True)
        self.debug.matrix(img_edges)
        self.debug.img(img_edges, 'image edges', gray=True)
        # segmap_edges = cv2.Canny(segmap, 10, 20, L2gradient=True)
        # self.debug.matrix(segmap_edges)
        # self.debug.img(segmap_edges, 'segmap_edges')

        print(labels[26].name)
        print(labels[26].trainId)

        car_map = np.where(segmap==labels[26].trainId, 255, 0)
        self.debug.img(car_map, 'car map', gray=True)

        out = np.where(car_map==255, 0, img_edges)
        self.debug.img(out, 'edge without car')

        self.debug.img(car_map, 'car map', gray=True)
        car_map_3c = np.stack([car_map for _ in range(3)], axis=-1)
        img_with_mask = np.where(car_map_3c==255, car_map_3c, img).astype(np.uint8)
        self.debug.img(img_with_mask, 'Image with Mask')
        self.debug.matrix(img_with_mask)

        # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
        # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
        # print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
        # print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

        # segfz_img = mark_boundaries(img, segments_fz)
        # segslic_img = mark_boundaries(img, segments_slic)
        # self.debug.img(segfz_img, 'Felzenszwalb Segmentation')
        # self.debug.img(segslic_img, 'SLIC Segmentation')

        # self.debug.matrix(segments_fz)


        segments_fz = felzenszwalb(img_with_mask, scale=100, sigma=0.5, min_size=50)
        segments_slic = slic(img_with_mask, n_segments=250, compactness=10, sigma=1)
        print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
        print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

        segfz_img = mark_boundaries(img_with_mask, segments_fz)
        segslic_img = mark_boundaries(img_with_mask, segments_slic)
        self.debug.img(segfz_img, 'Felzenszwalb Segmentation')
        self.debug.img(segslic_img, 'SLIC Segmentation')

        self.debug.matrix(segments_fz)
