import os
import torch
import matplotlib.pyplot as plt

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


class Debugger:
    def __init__(self, debug, save):
        self.debug = debug
        self.save = save
        self.order = 0

    def img(self, img, comment):
        if self.debug:
            print(comment)
            plt.figure(figsize=(10, 10), dpi=200)
            plt.imshow(img)
            # if len(list(img.shape)) == 2:
            #     plt.gray()
            plt.show()

            if self.save:
                filepath = os.path.join('../data/output', str(self.order) 
                        + '_' + comment[:5] + '.png')
                self._imsave(filepath, img)
                self.order += 1


    def param(self, string):
        if self.debug:
            print(string)


    def _imsave(self, path, img):
        if self.debug:
            cv2.imwrite(path, img)


    def matrix(self, mat):
        if self.debug:
            try:
                if len(mat.shape) == 1 or mat.shape[0] == 1:
                    print(mat)
                else:
                    print(mat.shape, mat.dtype, mat.min(), mat.mean(), mat.max())
            except:
                print(mat)
            print('-------------------')


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
