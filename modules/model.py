import cv2
import torch

class GANonymizer:
    def __init__(self, params, device, semseger, inpainter, shadow_detecter, debug):
        self.params = params
        self.devide = device
        self.semseger = semseger
        self.inpainter = inpainter
        self.shadow_detecter = shadow_detecter
        self.debug = debug
    
    
    def predict(self, img_path):
        # Loading input image
        img = cv2.imread(img_path)
        self.debug.img(img, 'Input Image')

        # semantic segmentation
        semseg_map = self._semseg(img)
        self.debug.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')

        # shadow detection
        # shadow_map = self._shadow_detect(img, semseg_map)
        # self.debug.img(shadow_map, 'Predicted Shadow Map')

        # create mask image
        # mask = self._get_mask(img, semseg_map, shadowmap)
        # self.debug.img(mask, 'Predicted Shadow Map')

        # inpainter
        
        

    def _semseg(self, img):
        pred = self.semseger(img)
        return pred

    
    def _inpainter(self, img):
        return None


    def _shadow_detect(self, img, segmap):
        return None


    def _get_mask(self, img, segmap, shadowmap):
        return None


    def _get_seg_boundary(img, segmap, shadowmap):
        return None
    

