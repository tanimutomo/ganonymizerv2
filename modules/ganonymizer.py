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
        img = np.array(img)
        self.debugger.matrix(img, 'Input Image')
        self.debugger.img(img, 'Input Image')

        # semantic segmentation
        semseg_map = self.semseger.process(img)
        self.debugger.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')
        # with open('./data/exp/segmap.pkl', mode='wb') as f:
        #     pickle.dump(semseg_map, f)
        # cv2.imwrite('data/exp/segmaps/segmap_' + img_path.split('/')[-1], semseg_map)

        # shadow detection
        # with open('./data/exp/segmap.pkl', mode='rb') as f:
        #     semseg_map = pickle.load(f)
        mask = self.shadow_detecter.detect(img, semseg_map, self.labels)
        self.debugger.matrix(img, 'img')
        self.debugger.matrix(mask, 'mask')

        # inpainter
        output = self.inpainter.inpaint(img, mask)
        self.debugger.matrix(output, 'output')
        self.debugger.img(output, 'inpainted image')
        
        

    def _get_mask(self, img, segmap, shadowmap):
        return None


    def _get_seg_boundary(img, segmap, shadowmap):
        return None
    

