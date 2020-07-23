import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import _split_train_val, create_dir
import torchvision.datasets as datasets
import torch.utils.data as utils
import errno
from PIL import Image


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Predicting with high correlation features')

    # Directories
    parser.add_argument('--data', type=str, default='datasets/',
                        help='location of the data corpus')
    parser.add_argument('--root_dir', type=str, default='default/',
                        help='root dir path to save the log and the final model')
    parser.add_argument('--save_dir', type=str, default='0/',
                        help='dir path (inside root_dir) to save the log and the final model')

    parser.add_argument('--load_dir', type=str, default='',
                        help='dir path (inside root_dir) to load model from')

    # Baseline (correlation based) method
    parser.add_argument('--beta', type=float, default=1,
                        help='coefficient for correlation based penalty')

    # adaptive batch norm
    parser.add_argument('--bn_eval', action='store_true',
                        help='adapt BN stats during eval')

    # dataset and architecture
    parser.add_argument('--dataset', type=str, default='fgbg_cmnist_cpr0.5-0.5',
                        help='dataset name')
    parser.add_argument('--arch', type=str, default='resnet',
                        help='arch name (resnet,cnn)')
    parser.add_argument('--depth', type=int, default=56,
                        help='number of resblocks if using resnet architecture')
    parser.add_argument('--k', type=int, default=1,
                        help='widening factor for wide resnet architecture')

    # Optimization hyper-parameters
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--bs', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--bn', action='store_true',
                        help='Use Batch norm')
    parser.add_argument('--noaffine', action='store_true',
                        help='no affine transformations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate ')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--init', type=str, default="he")
    parser.add_argument('--wdecay', type=float, default=0.0001,
                        help='weight decay applied to all weights')

    # meta specifications
    parser.add_argument('--validation', action='store_true',
                        help='Compute accuracy on validation set at each epoch')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--gpu', nargs='+', type=int, default=[0])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arg_parser()

    print('==> Preparing data..')
    data_dir_cmnist = args.data + 'cmnist/' + args.dataset + '/'

    data_x = np.load(data_dir_cmnist + 'train_x.npy')
    data_y = np.load(data_dir_cmnist + 'train_y.npy')

    for index, (i_data, label) in enumerate(zip(data_x, data_y)):
        # from matplotlib import cm

        i_data = np.rollaxis(i_data, 0, 3)
        # img = Image.fromarray(np.uint8(cm.gist_earth(i_data)*255),mode='RGB')
        img = Image.fromarray((i_data * 255.).astype('uint8'), mode='RGB')
        create_dir('./visualization/')
        img.save('./visualization/class_%d_%d_Train.png' % (label, index))
        if index >= 50:
            break

    data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
    data_y = torch.from_numpy(data_y).type('torch.LongTensor')

    test_data_x = np.load(data_dir_cmnist + 'test_x.npy')
    test_data_y = np.load(data_dir_cmnist + 'test_y.npy')

    for index, (i_data, label) in enumerate(zip(test_data_x, test_data_y)):
        # from matplotlib import cm

        i_data = np.rollaxis(i_data, 0, 3)
        # img = Image.fromarray(np.uint8(cm.gist_earth(i_data)*255),mode='RGB')
        img = Image.fromarray((i_data * 255.).astype('uint8'), mode='RGB')
        img.save('./visualization/class_%d_%d_Test.png' % (label, index))
        if index >= 50:
            break
