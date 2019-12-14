import argparse
import datetime
import distutils
import os 
import torch

from distutils.util import strtobool


def get_options(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # main
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--mode', type=str, default='exec', choices=['exec', 'debug', 'save'])
    parser.add_argument('--log_root', type=str, default='log')
    parser.add_argument('--gpu_id', type=int, default=0)

    # resize
    parser.add_argument('--resize_factor', type=float, default=None)
    
    # segmentation
    parser.add_argument('--semseg', type=str, default='DeepLabV3')
    parser.add_argument('--resnet', type=int, default=18)

    # inpaint
    parser.add_argument('--inpaint', type=str, default='EdgeConnect')
    parser.add_argument('--sigma', type=int, default=2, help='sigma of canny edge detection')

    # parse, print, and return
    opt = parser.parse_args(args)

    # set device
    opt.device = torch.device(
        'cuda:{}'.format(opt.gpu_id) if torch.cuda.is_available() else 'cpu'
    )
    # set log directory
    opt.log = os.path.join(
        opt.log_root,
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    )
    os.makedirs(opt.log)
    if opt.mode is not 'exec':
        opt.inter_log = os.path.join(opt.log, 'intermediates')
        os.mkdir(opt.inter_log)
    else:
        opt.inter_log = None

    _print_options(parser, opt)
    return opt


def _print_options(parser, opt):
    message = ''
    message += '---------------------------- Options --------------------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: {}]'.format(str(default))
        message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
    message += '---------------------------- End ------------------------------'
    print(message)

