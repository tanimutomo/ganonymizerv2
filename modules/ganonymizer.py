import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from .semantic_segmenter import SemanticSegmenter
from .shadow_detecter import ShadowDetecter
from .inpainter import Inpainter
from .utils import tensor_img_to_numpy

class GANonymizer:
    def __init__(self, config, device, labels, debugger):
        self.config = config
        self.devide = device
        self.labels = labels

        self.to_tensor = ToTensor()

        self.semseger = SemanticSegmenter(config, device, debugger)
        self.shadow_detecter = ShadowDetecter(debugger)
        self.inpainter = Inpainter(config, device, debugger)

        self.debugger = debugger
    
    
    def predict(self, img_path):
        # Loading input image
        img = Image.open(img_path)
        img = img.resize((int(img.width / 4), int(img.height / 4)))
        img = np.array(img)
        self.debugger.matrix(img, 'Input Image')
        self.debugger.img(img, 'Input Image')

        # semantic segmentation
        semseg_map = self.semseger.process(img)
        # with open('./data/exp/segmap.pkl', mode='wb') as f:
        #     pickle.dump(semseg_map, f)
        # with open('./data/exp/segmap.pkl', mode='rb') as f:
        #     semseg_map = pickle.load(f)
        self.debugger.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')

        # shadow detection
        mask = self.shadow_detecter.detect(img, semseg_map, self.labels)
        mask = self._expand_mask(mask)

        # resize
        self.debugger.matrix(img, 'img')
        self.debugger.matrix(mask, 'mask')
        # img = cv2.resize(img, (256, 256))
        # mask = cv2.resize(np.expand_dims(mask, axis=-1).astype(np.uint8),  (256, 256)).squeeze()

        # inpainter
        output, out_edge = self.inpainter.inpaint(img, mask)
        self.debugger.matrix(output, 'output')
        self.debugger.img(output, 'inpainted image')
        self.debugger.matrix(out_edge, 'out_edge')
        self.debugger.img(out_edge, 'inpainted edge')
        

    def _expand_mask(self, mask):
        width = int((mask.shape[0] + mask.shape[1]) / 256)
        print('width:', width)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = cv2.drawContours(mask, contours, -1, 255, width) 
        
        return mask


    def _get_seg_boundary(img, segmap, shadowmap):
        return None
    

