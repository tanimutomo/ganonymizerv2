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
        self.shadow_detecter = ShadowDetecter(config, debugger)
        self.inpainter = Inpainter(config, device, debugger)

        self.debugger = debugger
    
    
    def predict(self, img_path):
        img = self._load_img(img_path)
        segmap = self._semseg(img)
        if self.config['mask'] == 'entire':
            omask = self._entire_mask(img, segmap)
            smask = self._detect_shadow(img, omask)
            mask = self._combine_masks(omask, smask)
            self.debugger.img(mask, 'Mask (Object Mask + Shadow Mask)', main=True, gray=True)

            # image with mask
            mask_3c = np.stack([mask for _ in range(3)], axis=-1)
            img_with_mask = np.where(mask_3c==255, mask_3c, img).astype(np.uint8)
            self.debugger.img(img_with_mask, 'Image with Mask', main=True)

            out, _ = self._inpaint(img, mask)
            out = Image.fromarray(out)
            out.save('./data/exp/cityscapes_testset/{}_etr_out_shadow_{}.{}'.format(
                self.fname, self.config['shadow_mode'], self.fext))
        elif self.config['mask'] == 'separate':
            inputs = self._separated_mask(img, segmap)
            inpainteds = []
            for input in inputs:
                if input['area'] <= 400:
                    inpainted = cv2.inpaint(input['img'], input['mask'], 3, cv2.INPAINT_NS)
                else:
                    inpainted, inpainted_edge = self._inpaint(input['img'], input['mask'])
                inpainteds.append(inpainted)
            
            out = self._integrate_outputs(img, inputs, inpainteds)
            self.debugger.img(out, 'Final Output')
            out = Image.fromarray(out)
            out.save('./data/exp/cityscapes_testset/{}_sep_out_resized_{}.{}'.format(
                self.fname, self.config['resize_factor'], self.fext))


    def _load_img(self, img_path):
        # Loading input image
        print('===== Loading Image =====')
        self.fname, self.fext = img_path.split('/')[-1].split('.')
        img = Image.open(img_path)
        img = img.resize((int(img.width / self.config['resize_factor']),
            int(img.height / self.config['resize_factor'])))
        img = np.array(img)

        # visualization
        self.debugger.matrix(img, 'Input Image', main=True)
        self.debugger.img(img, 'Input Image', main=True)

        return img


    def _semseg(self, img):
        # semantic segmentation
        print('===== Semantic Segmentation =====')

        segmap_path = os.path.join(self.config['checkpoint'], self.fname + '_segmap.' + 'pkl')
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
        self.debugger.matrix(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3', main=True)
        self.debugger.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3', main=True)

        return semseg_map

    
    def _combine_masks(self, mask, smask):
        assert mask.shape == smask.shape
        return np.where(mask + smask > 0, 255, 0)


    def _entire_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Creating Mask Image =====')
        omask_path = os.path.join(self.config['checkpoint'], self.fname + '_omask.' + 'pkl')

        if self.config['mask_mode'] is 'exec':
            omask = self.shadow_detecter.mask(img, semseg_map, self.labels)
        elif self.config['mask_mode'] is 'pass':
            with open(omask_path, mode='rb') as f:
                omask = pickle.load(f)
        elif self.config['mask_mode'] is 'save':
            omask = self.shadow_detecter.mask(img, semseg_map, self.labels)
            with open(omask_path, mode='wb') as f:
                pickle.dump(omask, f)

        # visualization
        self.debugger.matrix(omask, 'Object Mask', main=True)
        self.debugger.img(omask, 'Object Mask', gray=True, main=True)

        return omask


    def _separated_mask(self, img, semseg_map):
        # create mask image and image with mask
        print('===== Create separated inputs =====')
        sep_inputs_path = os.path.join(self.config['checkpoint'], self.fname + '_sep_inputs.' + 'pkl')

        if self.config['mask_mode'] is 'exec':
            inputs = self.shadow_detecter.separated_mask(img, semseg_map, 
                    self.config['crop_rate'], self.labels)
        elif self.config['mask_mode'] is 'pass':
            with open(sep_inputs_path, mode='rb') as f:
                inputs = pickle.load(f)
        elif self.config['mask_mode'] is 'save':
            inputs = self.shadow_detecter.separated_mask(img, semseg_map, 
                    self.config['crop_rate'], self.labels)
            with open(sep_inputs_path, mode='wb') as f:
                pickle.dump(inputs, f)

        return inputs


    def _detect_shadow(self, img, mask):
        # shadow detection
        print('===== Shadow Detection =====')
        smask_path = os.path.join(self.config['checkpoint'], self.fname + '_smask.' + 'pkl')

        if self.config['shadow_mode'] is 'exec':
            smask = self.shadow_detecter.detect(img, mask)
        elif self.config['shadow_mode'] is 'pass':
            with open(smask_path, mode='rb') as f:
                smask = pickle.load(f)
        elif self.config['shadow_mode'] is 'save':
            smask = self.shadow_detecter.detect(img, mask)
            with open(smask_path, mode='wb') as f:
                pickle.dump(smask, f)
        elif self.config['shadow_mode'] is 'none':
            smask = np.zerso(mask.shape, dtype=np.uint8)

        # visualization
        self.debugger.matrix(smask, 'Shadow Mask', main=True)
        self.debugger.img(smask, 'Shadow Mask', gray=True, main=True)

        return smask


    def _inpaint(self, img, mask):
        # inpainter
        print('===== Image Inpainting and Edge Inpainting =====')
        inpaint_path = os.path.join(self.config['checkpoint'], self.fname + '_inpaint.' + 'pkl')
        inpaint_edge_path = os.path.join(self.config['checkpoint'], self.fname + '_inpaint_edge.' + 'pkl')

        if self.config['inpaint_mode'] is 'exec':
            inpainted, inpainted_edge = self.inpainter.inpaint(img, mask)
        elif self.config['inpaint_mode'] is 'pass':
            with open(inpaint_path, mode='rb') as f:
                inpainted = pickle.load(f)
            with open(inpaint_edge_path, mode='rb') as f:
                inpainted_edge = pickle.load(f)
        elif self.config['inpaint_mode'] is 'save':
            inpainted, inpainted_edge = self.inpainter.inpaint(img, mask)
            with open(inpaint_path, mode='wb') as f:
                pickle.dump(inpainted, f)
            with open(inpaint_edge_path, mode='wb') as f:
                pickle.dump(inpainted_edge, f)

        # visualization
        self.debugger.matrix(inpainted_edge, 'Inpainted Edge', main=True)
        self.debugger.img(inpainted_edge, 'Inpainted Edge', gray=True, main=True)
        self.debugger.matrix(inpainted, 'Inpainted Image', main=True)
        self.debugger.img(inpainted, 'Inpainted Image', main=True)

        return inpainted, inpainted_edge


    def _integrate_outputs(self, img, inputs, outputs):
        self.debugger.img(img, 'original image')
        for input, output in zip(inputs, outputs):
            box = input['box']
            origin_wh = (box[2] - box[0], box[3] - box[1])
            output = cv2.resize(output, origin_wh)
            mask_part = cv2.resize(input['mask'], origin_wh)
            self.debugger.matrix(output, 'output')
            self.debugger.img(output, 'output')

            out = np.zeros(img.shape, dtype=np.uint8)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            out[box[1]:box[3], box[0]:box[2], :] = output
            self.debugger.matrix(out, 'out')
            self.debugger.img(out, 'out')
            mask[box[1]:box[3], box[0]:box[2]] = mask_part
            
            mask = np.stack([mask for _ in range(3)], axis=-1)
            img = np.where(mask==255, out, img)
            self.debugger.img(img, 'img')

        return img


