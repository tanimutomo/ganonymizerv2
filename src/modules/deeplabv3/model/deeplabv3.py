# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from .resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from .aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    # def __init__(self, model_id, project_dir):
    def __init__(self, res_type, res_path, device):
        super(DeepLabV3, self).__init__()

        self.num_classes = 20
        self.device = device

        if res_type == 18:
            weight_name = "resnet18-5c106cde.pth"
            self.resnet = ResNet18_OS8(os.path.join(res_path, weight_name), torch.device('cpu')) # NOTE! specify the type of ResNet here
            self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        elif res_type == 34:
            weight_name = "resnet34-333f7ec4.pth"
            self.resnet = ResNet18_OS8(os.path.join(res_path, weight_name), torch.device('cpu'))
            self.aspp = ASPP(num_classes=self.num_classes)
        elif res_type == 50:
            weight_name = "resnet50-19c8e357.pth"
            self.resnet = ResNet18_OS50(os.path.join(res_path, weight_name), torch.device('cpu'))
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        elif res_type == 101:
            weight_name = "resnet101-5d3b4d8f.pth"
            self.resnet = ResNet18_OS101(os.path.join(res_path, weight_name), torch.device('cpu'))
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        elif res_type == 152:
            weight_name = "resnet152-b121ed2d.pth"
            self.resnet = ResNet18_OS152(os.path.join(res_path, weight_name), torch.device('cpu'))
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

        self.resnet.to(self.device)
        self.aspp.to(self.device)


    def forward(self, x):
        # image preprocessing
        x = self._preprocess(x)

        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        # image postprocess
        output = self._postprocess(output)

        return output

    
    def _preprocess(self, img):
        # normalize the img (with mean and std for the pretrained ResNet):
        img = img / 255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        img = torch.unsqueeze(img, 0) # (shape: (batch_size, 3, img_h, img_w))
        img = img.to(self.device) # (shape: (batch_size, 3, img_h, img_w))

        return img


    def _postprocess(self, img):
        # (shape: (batch_size, num_classes, img_h, img_w))
        img = torch.squeeze(img, 0) # (shape: (num_classes, img_h, img_w))
        img = img.data.cpu().numpy() # (shape: (num_classes, img_h, img_w))
        pred_labels = np.argmax(img, axis=0) # (shape: (img_h, img_w))
        pred_labels = pred_labels.astype(np.uint8)

        labels = np.reshape(pred_labels, -1)

        return pred_labels

