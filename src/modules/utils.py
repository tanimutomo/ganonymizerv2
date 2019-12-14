import collections
import cv2
import numpy as np
import os
import PIL
import torch
import torchvision

from PIL import Image
from collections import namedtuple
from torchvision import transforms


def demo_resize(img):
    # Resize Input image for demonstration
    h, w = img.shape[0], img.shape[1]
    if h <= 1024 and w <= 1024:
        return img

    side, new_side = np.array([w, h]), np.array([0, 0])
    if side[0] >= side[1]:
        is_side_exchanged = False
    else:
        side = side[::-1]
        is_side_exchanged = True

    new_side[0] = 1024
    new_side[1] = side[1] * (1024 / side[0])
    new_side = new_side.astype(np.int32)
    new_side[1] -= (new_side[1] % 4)

    if is_side_exchanged:
        new_side = new_side[::-1]
    out = Image.fromarray(img)
    out = out.resize((new_side[0], new_side[1]))

    return np.array(out).astype(np.uint8)


class Debugger:
    def __init__(self, mode, save_dir):
        self.mode = mode
        self.save_dir = save_dir
        self.to_pil = transforms.ToPILImage()

    def value(self, value, comment):
        if self.mode == 'debug':
            print('-----', comment, '-----')
            print(value)
            print('-' * (len(comment) + 12))

    def matrix(self, mat, comment):
        if self.mode == 'debug':
            print('-----', comment, '-----')
            if type(mat) is torch.Tensor:
                if 'float' in str(mat.dtype):
                    print('shape: {}   dtype: {}   min: {}   mean: {}   median: {}   max: {}   device: {}'.format(
                        mat.shape, mat.dtype, mat.min(), mat.mean(), mat.median(), mat.max(), mat.device))
                else:
                    print('shape: {}   dtype: {}   min: {}   mean: {}   median: {}   max: {}   device: {}'.format(
                        mat.shape, mat.dtype, mat.min(), mat.to(torch.float32).mean(), mat.median(), mat.max(), mat.device))

            elif type(mat) is np.ndarray:
                if mat.ndim == 1:
                    print(mat)
                else:
                    print('shape: {}   dtype: {}   min: {}   mean: {}   median: {}   max: {}'.format(
                        mat.shape, mat.dtype, mat.min(), mat.mean(), mat.median(), mat.max()))
            else:
                print('[Warning] Input type is {}, not matrix(numpy.ndarray or torch.tensor)!'.format(
                    type(mat)))
                print(mat)
            print('-' * (len(comment) + 12))

    def imsave(self, img, filename):
        if self.mode == 'save':
            path = os.path.join(self.save_dir, filename)
            if type(img) is torch.Tensor:
                img = self.to_pil(img)
                img.save(path)
            elif type(img) is np.ndarray:
                img = Image.fromarray(img)
                img.save(path)
            else:
                raise RuntimeError('The type of input image must be numpy.ndarray or torch.Tensor.')


# function for colorizing a label image:
def label_img_to_color(img):
    label_data = [
            {'color': [128, 64,128], 'name': 'road'},
            {'color': [244, 35,232], 'name': 'sidewalk'},
            {'color': [ 70, 70, 70], 'name': 'building'},
            {'color': [102,102,156], 'name': 'wall'},
            {'color': [190,153,153], 'name': 'fence'},
            {'color': [153,153,153], 'name': 'pole'},
            {'color': [250,170, 30], 'name': 'traffic light'},
            {'color': [220,220,  0], 'name': 'traffic sign'},
            {'color': [107,142, 35], 'name': 'vegetation'},
            {'color': [152,251,152], 'name': 'terrain'},
            {'color': [ 70,130,180], 'name': 'sky'},
            {'color': [220, 20, 60], 'name': 'person'},
            {'color': [255,  0,  0], 'name': 'rider'},
            {'color': [  0,  0,142], 'name': 'car'},
            {'color': [  0,  0, 70], 'name': 'truck'},
            {'color': [  0, 60,100], 'name': 'bus'},
            {'color': [  0, 80,100], 'name': 'train'},
            {'color': [  0,  0,230], 'name': 'motorcycle'},
            {'color': [119, 11, 32], 'name': 'bicycle'},
            {'color': [81,  0,  81], 'name': 'others'}
            ]

    img_height, img_width = img.shape
    img3c = np.stack([img, img, img], axis=-1).astype(np.int32)
    lc_img = np.zeros((800, 1050, 3), dtype=np.uint8)
    
    for label, data in enumerate(label_data):
        img3c = np.where(img3c == [label, label, label], data['color'], img3c)
        x, y = label % 4, int((label - (label % 4)) / 4)
        tlx, tly = 50 + x * 250, 50 + y * 150
        lc_img = cv2.rectangle(lc_img, (tlx, tly), (tlx + 200, tly + 100),
                data['color'], thickness=-1)
        lc_img = cv2.putText(lc_img, data['name'], (tlx + 10, tly + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return img3c.astype(np.uint8), lc_img


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
