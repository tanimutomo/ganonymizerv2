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
        # self.debug.img(out, 'edge without car')

        car_map_3c = np.stack([car_map for _ in range(3)], axis=-1)
        img_with_mask = np.where(car_map_3c==255, car_map_3c, img).astype(np.uint8)
        self.debug.img(img_with_mask, 'Image with Mask')
        self.debug.matrix(img_with_mask)

        # oritginal image superpixel
        # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
        # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
        # print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
        # print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

        # segfz_img = mark_boundaries(img, segments_fz)
        # segslic_img = mark_boundaries(img, segments_slic)
        # self.debug.img(segfz_img, 'Felzenszwalb Segmentation')
        # self.debug.img(segslic_img, 'SLIC Segmentation')

        # self.debug.matrix(segments_fz)


        # image with mask superpixel
        self._slic(img_with_mask)


        # calcurate the bouding box
        car_map = car_map.astype(np.uint8)
        self.debug.matrix(car_map)
        contours, _ = cv2.findContours(car_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        out1 = car_map.copy()
        thresh = 10000
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            out1 = cv2.rectangle(out1, (x, y), (x+w, y+h), 255, 2)
            if w*h >= thresh:
                bboxes.append([x, y, x+w, y+h])

        self.debug.img(out1, 'car map with bbox')


        # extract bottom part of the image with mask
        H, W = car_map.shape[0], car_map.shape[1]
        print(W, H)
        for box in bboxes:
            half_w, half_h = int((box[2] - box[0]) / 2), int((box[3] - box[1]) / 2)
            box[0] = box[0] - half_w if box[0] - half_w >= 0 else 0
            # box[1] = box[1] - half_h if box[1] - half_h >= 0 else 0
            box[2] = box[2] + half_w if box[2] + half_w < W else W - 1
            box[3] = box[3] + half_h if box[3] + half_h < H else H - 1

        out2 = car_map.copy()
        for box in bboxes:
            out2 = cv2.rectangle(out2, (box[0], box[1]), (box[2], box[3]), 255, 2)

        self.debug.img(out2, 'expanded bounding boxes')

        cars = []
        for box in bboxes:
            car = img_with_mask[box[1]:box[3], box[0]:box[2], :]
            self.debug.img(car, 'car part')
            cars.append(car)

        for car in cars:
            self._slic(car)


    def _slic(self, img):
        segments = img.shape[0] * img.shape[1] / 500
        segments_slic = slic(img, n_segments=segments, compactness=10, sigma=1)
        print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

        segslic_img = mark_boundaries(img, segments_slic)
        self.debug.img(segslic_img, 'SLIC Segmentation')
        self.debug.matrix(segments_slic) 

