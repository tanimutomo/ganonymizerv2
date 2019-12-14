import os
import cv2
import torch
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple
from skimage.segmentation import mark_boundaries


def where(cond, true_val, false_val):
    true_tensor = torch.ones_like(cond).to(cond.device) * true_val
    false_tensor = torch.ones_like(cond).to(cond.device) * true_val
    return torch.where(cond, true_tensor, false_tensor)


def load_img(img_path):
    # Loading input image
    print('===== Loading Image =====')
    fname, fext = img_path.split('/')[-1].split('.')
    img = Image.open(img_path)
    return img, fname, fext


def load_video(path):
    """
    Load the input video data
    """
    print('===== Loading Image =====')
    fname, fext = path.split('/')[-1].split('.')
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))

    return fname, cap, fps, frames, W, H
    

def video_writer(in_path, fps, width, height):
    fname, fext = in_path.split('/')[-1].split('.')
    out_path = os.path.join('/'.join(in_path.split('/')[:-2]),
                            "output", "out_" + fname + '.' + fext)
    print("Saved Video Path:", out_path)
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    return writer


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


def tensor_img_to_numpy(tensor):
    array = tensor.numpy()
    array = np.transpose(array, (1, 2, 0))
    return array


def detect_object(img, debugger):
    num, labelmap, stats, _ = cv2.connectedComponentsWithStats(img)
    debugger.img(labelmap, 'labelmap')
    # stat is [tl_x, tl_y, w, h, area]
    label_list = [i+1 for i in range(num - 1)]
    return labelmap, stats[1:], label_list


def expand_mask(mask, width):
    if width == 0: return mask
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, 255, width) 
    return mask.astype(np.uint8)


def write_labels(img, segmap, size):
    out = mark_boundaries(img, segmap)
    labels = np.unique(segmap)
    for label in labels:
        ys, xs = np.where(segmap==label)
        my, mx = np.median(ys).astype(np.int32), np.median(xs).astype(np.int32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        out = cv2.putText(out, str(label), (mx - 10, my), font, size, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def bar_plot(gray, label, comment, cluster=None):
    print('-'*10, comment, '-'*10)
    plt.figure(figsize=(10, 10), dpi=200)
    if cluster is None:
        plt.bar(np.arange(gray.shape[0]), gray,
                tick_label=label, align='center')
    else:
        color = flatten([['r', 'g', 'b', 'c', 'm', 'y'] for _ in range(100)])
        bar_color = [color[c] for c in cluster]
        plt.bar(np.arange(gray.shape[0]), gray,
                color=bar_color, tick_label=label, align='center')
    plt.show()
    print('-' * (len(comment) + 22))


def flatten(nested_list):
    """2重のリストをフラットにする関数"""
    return [e for inner_list in nested_list for e in inner_list]


def pmd_mode_change(config, pmd):
    if pmd == 'on':
        config.main_mode = 'exec'
        config.semseg_mode = 'save'
        config.mask_mode = 'save'
        config.split_mode = 'save'
        config.shadow_mode = 'save'
        config.random_mode = 'save'
        config.divide_mode = 'exec'
        config.inpaint_mode = 'exec'
    elif pmd == 'off':
        config.main_mode = 'exec'
        config.semseg_mode = 'pass'
        config.mask_mode = 'pass'
        config.split_mode = 'pass'
        config.shadow_mode = 'pass'
        config.random_mode = 'pass'
        config.divide_mode = 'none'
        config.inpaint_mode = 'exec'
    return config


def demo_config(config):
    config.main_mode = 'exec'
    config.semseg_mode = 'exec'
    config.mask_mode = 'exec'
    config.split_mode = 'exec'
    config.shadow_mode = 'exec'
    config.divide_mode = 'exec'
    config.inpaint_mode = 'exec'

    config.random_mode = 'none'

    return config


def create_dir(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            dirpath = os.path.dirname(path)
            rootpath = os.path.dirname(dirpath)
            if os.path.basename(dirpath) == 'input':
                _create_det_dir(rootpath)
                return path
            else:
                os.mkdir(os.path.join(dirpath, 'input'))
                files = os.listdir(dirpath)
                files = [os.path.join(dirpath, f) for f in files 
                        if os.path.isfile(os.path.join(dirpath, f)) and f[0] != '.']
                for f in files:
                    shutil.move(f, os.path.join(dirpath, 'input'))
                _create_det_dir(dirpath)
                return os.path.join(dirpath, 'input', os.path.basename(path))

        elif os.path.isdir(path):
            if os.path.basename(path) == 'input':
                raise RuntimeError('path should be dataset root '\
                                   'directory path os an image path')
            else:
                if not os.path.exists(os.path.join(path, 'input')):
                    os.mkdir(os.path.join(path, 'input'))
                files = os.listdir(path)
                files = [os.path.join(path, f) for f in files 
                        if os.path.isfile(os.path.join(path, f)) and f[0] != '.']
                for f in files:
                    shutil.move(f, os.path.join(path, 'input'))
                _create_det_dir(path)
                return path
    else:
        raise RuntimeError('input argument path is not existed')


def _create_det_dir(rootpath):
    for name in ['ckpt', 'output', 'pmd']:
        if not os.path.exists(os.path.join(rootpath, name)):
            os.mkdir(os.path.join(rootpath, name))


class Config(dict):
    def __init__(self, config):
        self._conf = config
 
    def __getattr__(self, name):
        if self._conf.get(name) is not None:
            return self._conf[name]

        return None


class Debugger:
    def __init__(self, mode, save_dir=None):
        self.mode = mode
        self.save_dir = save_dir

    def img(self, img, comment):
        if self.mode is 'debug':
            print('-'*10, comment, '-'*10)
            self.matrix(img, comment)
            if type(img) is torch.Tensor and len(list(img.shape)) == 3:
                img = tensor_img_to_numpy(img)
                
            plt.figure(figsize=(10, 10), dpi=200)
            plt.imshow(img)
            if img.ndim == 2 and np.unique(img).size == 2:
                plt.gray()
            plt.show()
            print('-' * (len(comment) + 22))


    def param(self, param, comment):
        if self.mode is 'debug':
            print('-----', comment, '-----')
            print(param)
            print('-' * (len(comment) + 12))


    def imsave(self, img, filename):
        if self.mode in ['debug', 'save']:
            path = os.path.join(self.save_dir, filename)
            if type(img) is torch.Tensor:
                img = img.cpu().numpy().astype(np.uint8)
                img = Image.fromarray(img)
                img.save(path)
            elif type(img) is np.ndarray:
                img = Image.fromarray(img)
                img.save(path)
            else:
                raise RuntimeError('The type of input image must be numpy.ndarray or torch.Tensor.')


    def matrix(self, mat, comment):
        if self.mode is 'debug':
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
