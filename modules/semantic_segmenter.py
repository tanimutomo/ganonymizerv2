import os
import torch
from .deeplabv3.model.deeplabv3 import DeepLabV3

class SemanticSegmenter():
    def __init__(self, config, device, debugger):
        self.network = config['semseg_net']
        self.resnet = config['resnet']
        self.device = device
        self.debugger = debugger


    def process(self, img)
        # semantic segmentation
        if self.network == 'DeepLabV3':
            model = self._set_deeplabv3()
        else:
            raise RuntimeError('Please prepare {} model in modules/ .'.format(self.network))
        semseg_map = model(img)
        self.debugger.img(semseg_map, 'Semantic Segmentation Map Prediction by DeepLabV3')
        with open('./data/exp/segmap.pkl', mode='wb') as f:
            pickle.dump(semseg_map, f)
        # cv2.imwrite('data/exp/segmaps/segmap_' + img_path.split('/')[-1], semseg_map)

        return semseg_map


    def _set_deeplabv3(self, resnet):
        weights = os.path.join(os.getcwd(),
                'modules/deeplabv3/pretrained/model_13_2_2_2_epoch_580.pth')
        resnet_root = os.path.join(os.getcwd(),
            'modules/deeplabv3/pretrained/resnet')
        model = DeepLabV3(self.resnet, resnet_root, self.device)
        param = torch.load(weights, map_location=self.device)
        model.load_state_dict(param)
        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

        return model
