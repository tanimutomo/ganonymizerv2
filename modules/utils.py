import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

def set_networks(DeepLabV3, resnet, device):
    #Set up neural networks
    weights = os.path.join(os.getcwd(),
            'modules/deeplabv3/pretrained/model_13_2_2_2_epoch_580.pth')
    resnet_root = os.path.join(os.getcwd(),
        'modules/deeplabv3/pretrained/resnet')
    semseger = DeepLabV3(resnet, resnet_root, device)
    param = torch.load(weights, map_location=device)
    semseger.load_state_dict(param)
    semseger.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

    return semseger


def tensor_img_to_numpy(tensor):
    array = tensor.numpy()
    array = np.transpose(array, (1, 2, 0))
    return array


class Debugger:
    def __init__(self, main, debug, save=False, output_dir=None):
        self.main = main
        self.debug = debug
        self.save = save
        self.order = 0
        self.output_dir = output_dir

    def img(self, img, comment, gray=False, main=False):
        if self.debug or (self.main and main):
            print(comment)
            if type(img) is torch.Tensor and len(list(img.shape)) == 3:
                img = tensor_img_to_numpy(img)
                
            plt.figure(figsize=(10, 10), dpi=200)
            plt.imshow(img)
            if gray:
                plt.gray()
            plt.show()

            if self.save:
                if self.output_dir is None:
                    raise RuntimeError('Please specify output directory for saving images')
                else:
                    filepath = os.path.join(self.output_dir, str(self.order) 
                            + '_' + comment[:5] + '.png')
                    self.imsave(img, filepath)
                    self.order += 1


    def param(self, param, comment, main=False):
        if self.debug or (self.main and main):
            print('-----', comment, '-----')
            print(param)
            print('-' * (len(comment) + 12))


    def imsave(self, img, path, main=False):
        if self.debug or (self.main and main):
            print(img.shape)
            if type(img) is torch.Tensor:
                img = img.cpu().numpy().astype(np.uint8)
                img = Image.fromarray(img)
                img.save(path)
            elif type(img) is np.ndarray:
                cv2.imwrite(path, img)
            else:
                raise RuntimeError('The type of input image must be numpy.ndarray or torch.Tensor.')


    def matrix(self, mat, comment, main=False):
        if self.debug or (self.main and main):
            print('-----', comment, '-----')
            if type(mat) is torch.Tensor:
                if 'float' in str(mat.dtype):
                    print('shape: {}   dtype: {}   min: {}   mean: {}   max: {}   device: {}'.format(
                        mat.shape, mat.dtype, mat.min(), mat.mean(), mat.max(), mat.device))
                else:
                    print('shape: {}   dtype: {}   min: {}   mean: {}   max: {}   device: {}'.format(
                        mat.shape, mat.dtype, mat.min(), mat.to(torch.float32).mean(), mat.max(), mat.device))

            elif type(mat) is np.ndarray:
                if mat.shape[0] == 0:
                    print(mat)
                else:
                    print('shape: {}   dtype: {}   min: {}   mean: {}   max: {}'.format(
                        mat.shape, mat.dtype, mat.min(), mat.mean(), mat.max()))
            else:
                print('[Warning] Input type is {}, not matrix(numpy.ndarray or torch.tensor)!'.format(
                    type(mat)))
                print(mat)
            print('-' * (len(comment) + 12))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      19 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      19 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      19 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      19 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      19 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      19 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      19 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      19 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      19 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       19 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]
