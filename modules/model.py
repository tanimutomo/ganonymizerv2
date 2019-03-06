import cv2
import torch
import pickle
from PIL import Image
from .semantic_segmenter import SemanticSegmenter
from .shadow_detecter import ShadowDetecter

class GANonymizer:
    def __init__(self, config, device, labels, debugger):
        self.config = config
        self.devide = device

        self.semseger = SemanticSegmenter(config, device, debugger)
        self.shadow_detecter = ShadowDetecter(debugger)
        # self.inpainter = inpainter

        self.labels = labels
        self.debugger = debugger
    
    
    def predict(self, img_path):
        # Loading input image
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
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
        shadow_map = self.shadow_detecter.detect(img, semseg_map, self.labels)
        # self.debugger.img(shadow_map, 'Predicted Shadow Map')

        # create mask image
        # mask = self._get_mask(img, semseg_map, shadowmap)
        # self.debugger.img(mask, 'Predicted Shadow Map')

        # inpainter
        
        

    def _get_mask(self, img, segmap, shadowmap):
        return None


    def _get_seg_boundary(img, segmap, shadowmap):
        return None
    

