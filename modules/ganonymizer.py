import os
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
        print('\n', '===== Loading Image =====')
        fname, fext = img_path.split('/')[-1].split('.')
        img = Image.open(img_path)
        img = img.resize((int(img.width / 4), int(img.height / 4)))
        img = np.array(img)

        # visualization
        self.debugger.matrix(img, 'Input Image')
        self.debugger.img(img, 'Input Image')

        # semantic segmentation
        print('\n', '===== Semantic Segmentation =====')

        segmap_path = os.path.join(self.config['checkpoint'], fname + '_segmap' + 'pkl')
        if self.config['semseg_mode'] is 'exec':
            semseg_map = self.semseger.process(img)
        elif self.config['semseg_mode'] is 'pass':
            with open(segmap_path, mode='rb') as f:
                semseg_map = pickle.load(f)
        elif self.config['semseg_mode'] is 'save':
            semseg_map = self.semseger.process(img)
            with open(segmap_path, mode='wb') as f:
                pickle.dump(semseg_map, f)

        # visualization
        self.debugger.matrix(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')
        self.debugger.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')


        # create mask image and image with mask
        print('\n', '===== Creating Mask Image =====')
        mask_path = os.path.join(self.config['checkpoint'], fname + '_mask' + 'pkl')
        img_with_mask_path = os.path.join(self.config['checkpoint'], fname + '_img_with_mask' + 'pkl')

        if self.config['mask_mode'] is 'exec':
            mask, img_with_mask = self.shadow_detecter.mask(img, semseg_map, self.labels)
        elif self.config['mask_mode'] is 'pass':
            with open(mask_path, mode='rb') as f:
                mask = pickle.load(f)
            with open(img_with_mask_path, mode='rb') as f:
                img_with_mask = pickle.load(f)
        elif self.config['mask_mode'] is 'save':
            mask, img_with_mask = self.shadow_detecter.mask(img, semseg_map, self.labels)
            with open(mask_path, mode='wb') as f:
                pickle.dump(mask, f)
            with open(img_with_mask_path, mode='wb') as f:
                pickle.dump(img_with_mask, f)

        # visualization
        self.debugger.matrix(mask, 'mask')
        self.debugger.img(mask, 'mask', gray=True)
        self.debugger.matrix(img_with_mask, 'img_with_mask')
        self.debugger.img(img_with_mask, 'img_with_mask')


        # shadow detection
        print('\n', '===== Shadow Detection =====')
        self.shadow_detecter.detect(img, mask, img_with_mask)


        # inpainter
        print('\n', '===== Image Inpainting and Edge Inpainting =====')
        inpaint_path = os.path.join(self.config['checkpoint'], fname + '_inpaint' + 'pkl')
        inpaint_edge_path = os.path.join(self.config['checkpoint'], fname + '_inpaint_edge' + 'pkl')

        if self.config['inpaint_mode'] is 'exec':
            inpainted, inpainted_edge = self.inpainter.inpaint(img_with_mask, mask)
        elif self.config['inpaint_mode'] is 'pass':
            with open(inpaint_path, mode='rb') as f:
                inpainted = pickle.load(f)
            with open(inpaint_edge_path, mode='rb') as f:
                inpainted_edge = pickle.load(f)
        elif self.config['inpaint_mode'] is 'save':
            inpainted, inpainted_edge = self.inpainter.inpaint(img_with_mask, mask)
            with open(inpaint_path, mode='wb') as f:
                pickle.dump(inpainted, f)
            with open(inpaint_edge_path, mode='wb') as f:
                pickle.dump(inpainted_edge, f)

        # visualization
        self.debugger.matrix(inpainted_edge, 'Inpainted Edge')
        self.debugger.img(inpainted_edge, 'Inpainted Edge', gray=True)
        self.debugger.matrix(inpainted, 'Inpainted Image')
        self.debugger.img(inpainted, 'Inpainted Image')
        

