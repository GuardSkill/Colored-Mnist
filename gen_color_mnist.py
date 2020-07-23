import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import tqdm
from torch.autograd import Variable
import argparse
import os

from datasets.mnist import MNIST


def get_color_codes(cpr):
    C = np.random.rand(len(cpr), nb_classes, 3)
    C = C / np.max(C, axis=2)[:, :, None]
    print(C.shape)  # (2, 10, 3) random matrix (0-1 in axis 2)
    return C


def gen_fgbgcolor_data(loader, img_size=(3, 28, 28), cpr=[0.5, 0.5], noise=10.):
    if cpr is not None:
        assert sum(cpr) == 1, '--cpr must be a non-negative list which sums to 1'
        Cfg = get_color_codes(cpr)  # (2, 10, 3) random matrix (0-1 in axis 2)
        Cbg = get_color_codes(cpr)
    else:
        Cfg = get_color_codes([1])
        Cbg = get_color_codes([1])
    tot_iters = len(loader)
    for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
        x, targets = next(iter(loader))
        assert len(
            x.size()) == 4, 'Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)'
        targets = targets.cpu().numpy()
        bs = targets.shape[0]

        x = (((x * 255) > 150) * 255).type('torch.FloatTensor')  # binary the map  shape=2000.3,28,28   value: 0/255
        x_rgb = torch.ones(x.size(0), 3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
        x_rgb = x_rgb * x
        x_rgb_fg = 1. * x_rgb  # 2000.3,28,28
        # np.random.multinomial(1, cpr, targets.shape[0])----> shape=(M,2)   M=targets.shape[0] 2= cpr.size()
        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]),  # 多项式分布
                                 axis=1) if cpr is not None else 0  # shape=(M,)
        c = Cfg[color_choice, targets] if cpr is not None else Cfg[
            color_choice, np.random.randint(nb_classes, size=targets.shape[
                0])]  # 2000,3  choose the random [0,1] and special class colored by a color
        c = c.reshape(-1, 3, 1, 1)  # 2000,3,1,1
        c = torch.from_numpy(c).type('torch.FloatTensor')
        x_rgb_fg[:, 0] = x_rgb_fg[:, 0] * c[:, 0]
        x_rgb_fg[:, 1] = x_rgb_fg[:, 1] * c[:, 1]
        x_rgb_fg[:, 2] = x_rgb_fg[:, 2] * c[:, 2]

        bg = (255 - x_rgb)  # invert the original x_rgb  (to color the background)
        # c = C[targets] if np.random.rand()>cpr else C[np.random.randint(C.shape[0], size=targets.shape[0])]
        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1) if cpr is not None else 0
        c = Cbg[color_choice, targets] if cpr is not None else Cbg[
            color_choice, np.random.randint(nb_classes, size=targets.shape[0])]
        c = c.reshape(-1, 3, 1, 1)
        c = torch.from_numpy(c).type('torch.FloatTensor')
        bg[:, 0] = bg[:, 0] * c[:, 0]
        bg[:, 1] = bg[:, 1] * c[:, 1]
        bg[:, 2] = bg[:, 2] * c[:, 2]
        x_rgb = x_rgb_fg + bg
        x_rgb = x_rgb + torch.tensor((noise) * np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
        x_rgb = torch.clamp(x_rgb, 0., 255.)
        if i == 0:
            color_data_x = np.zeros((bs * tot_iters, *img_size))  # initiate these images
            color_data_y = np.zeros((bs * tot_iters,))
        color_data_x[i * bs: (i + 1) * bs] = x_rgb / 255.
        color_data_y[i * bs: (i + 1) * bs] = targets
    return color_data_x, color_data_y


def gen_foreground_color_data(loader, img_size=(3, 28, 28), cpr=[0.5, 0.5], noise=10.):
    if cpr is not None:
        assert sum(cpr) == 1, '--cpr must be a non-negative list which sums to 1'
        Cfg = get_color_codes(cpr)  # (2, 10, 3) random matrix (0-1 in axis 2)
        Cbg = get_color_codes(cpr)
    else:
        Cfg = get_color_codes([1])
        Cbg = get_color_codes([1])
    tot_iters = len(loader)
    for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
        x, targets = next(iter(loader))
        assert len(
            x.size()) == 4, 'Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)'
        targets = targets.cpu().numpy()
        bs = targets.shape[0]

        x = (((x * 255) > 150) * 255).type('torch.FloatTensor')  # binary the map  shape=2000.3,28,28   value: 0/255
        x_rgb = torch.ones(x.size(0), 3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
        x_rgb = x_rgb * x
        x_rgb_fg = 1. * x_rgb  # 2000.3,28,28
        # np.random.multinomial(1, cpr, targets.shape[0])----> shape=(M,2)   M=targets.shape[0] 2= cpr.size()
        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]),  # 多项式分布
                                 axis=1) if cpr is not None else 0  # shape=(M,)
        c = Cfg[color_choice, targets] if cpr is not None else Cfg[
            color_choice, np.random.randint(nb_classes, size=targets.shape[
                0])]  # 2000,3  choose the random [0,1] and special class colored by a color
        c = c.reshape(-1, 3, 1, 1)  # 2000,3,1,1
        c = torch.from_numpy(c).type('torch.FloatTensor')
        x_rgb_fg[:, 0] = x_rgb_fg[:, 0] * c[:, 0]
        x_rgb_fg[:, 1] = x_rgb_fg[:, 1] * c[:, 1]
        x_rgb_fg[:, 2] = x_rgb_fg[:, 2] * c[:, 2]

        x_rgb = x_rgb_fg + torch.tensor((noise) * np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
        x_rgb = torch.clamp(x_rgb, 0., 255.)
        if i == 0:
            color_data_x = np.zeros((bs * tot_iters, *img_size))  # initiate these images
            color_data_y = np.zeros((bs * tot_iters,))
        color_data_x[i * bs: (i + 1) * bs] = x_rgb / 255.
        color_data_y[i * bs: (i + 1) * bs] = targets
    return color_data_x, color_data_y


if __name__ == '__main__':
    data_path = 'datasets/'

    parser = argparse.ArgumentParser(description='Generate colored MNIST')

    # Hyperparams
    parser.add_argument('--cpr', nargs='+', type=float, default=[0.5, 0.5],
                        help='color choice is made corresponding to a class with these probability')
    parser.add_argument('--seed', type=int, default=18,
                        help='The seed for some random operations')
    parser.add_argument('--only_foreground', type=int, default=0,
                        help='The seed for some random operations')
    args = parser.parse_args()

    # Fix the seed to make the results is re-implementable
    seed = args.seed
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trans = ([transforms.ToTensor()])
    trans = transforms.Compose(trans)
    fulltrainset = MNIST(root=data_path, train=True, download=False, transform=trans)
    trainloader = torch.utils.data.DataLoader(fulltrainset, batch_size=2000, shuffle=False, num_workers=2,
                                              pin_memory=True)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=trans)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
    nb_classes = 10
    if args.only_foreground:
        dir_name = data_path + 'cmnist/' + 'fg_cmnist_cpr' + '-'.join(str(p) for p in args.cpr) + '/'
    else:
        dir_name = data_path + 'cmnist/' + 'fgbg_cmnist_cpr' + '-'.join(str(p) for p in args.cpr) + '/'
    print(dir_name)
    if not os.path.exists(data_path + 'cmnist/'):
        os.mkdir(data_path + 'cmnist/')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if args.only_foreground:
        color_data_x, color_data_y =gen_foreground_color_data(trainloader, img_size=(3, 28, 28), cpr=args.cpr, noise=10.)
    else:
        color_data_x, color_data_y = gen_fgbgcolor_data(trainloader, img_size=(3, 28, 28), cpr=args.cpr, noise=10.)
    np.save(dir_name + '/train_x.npy', color_data_x)
    np.save(dir_name + '/train_y.npy', color_data_y)

    if args.only_foreground:
        color_data_x, color_data_y =gen_foreground_color_data(testloader, img_size=(3, 28, 28), cpr=None, noise=10.)
    else:
        color_data_x, color_data_y = gen_fgbgcolor_data(testloader, img_size=(3, 28, 28), cpr=None, noise=10.)
    np.save(dir_name + 'test_x.npy', color_data_x)
    np.save(dir_name + 'test_y.npy', color_data_y)

# generate color codes
