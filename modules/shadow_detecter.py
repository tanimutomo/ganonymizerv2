import cv2
import numpy as np
import torch

class ShadowDetecter:
    def __init__(self, debug):
        self.debug = debug

    
    def detect(self, img, segmap):
        img_edges = cv2.Canny(img, 100, 200, L2gradient=True)
        self.debug.matrix(img_edges)
        self.debug.img(img_edges, 'image edges')
        segmap_edges = cv2.Canny(segmap, 10, 20, L2gradient=True)
        self.debug.matrix(segmap_edges)
        self.debug.img(segmap_edges, 'segmap_edges')


