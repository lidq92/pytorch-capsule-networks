import os
import torch
import torchvision
import torchvision.transforms as transforms
from trainer import CapsNetTrainer
import yaml, argparse
from utils.util import ensure_dir
from logger.logger import Logger  #


def main(args):
    conf = yaml.load(open(args.config))
    conf.update(conf[conf['model']])

    if args.multi_gpu:
        conf['batch_size'] *= torch.cuda.device_count()

    datasets = {
        'MNIST': torchvision.datasets.MNIST,
        'CIFAR': torchvision.datasets.CIFAR10
    }
    if conf['dataset'].upper() == 'MNIST':
        conf['data_path'] = os.path.join(conf['data_path'], 'MNIST')
        size = 28
        classes = list(range(10))
        mean, std = ((0.1307,), (0.3081,))
    elif conf['dataset'].upper() == 'CIFAR':
        conf['data_path'] = os.path.join(conf['data_path'], 'CIFAR')
        size = 32
        classes = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        raise ValueError('Dataset must be either MNIST or CIFAR!')
    transform = transforms.Compose([
        transforms.RandomCrop(size, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    loaders = {}
    trainset = datasets[conf['dataset'].upper()](root=conf['data_path'],
                        train=True, download=True, transform=transform)
    testset = datasets[conf['dataset'].upper()](root=conf['data_path'],
                       train=False, download=True, transform=transform)
    loaders['train'] = torch.utils.data.DataLoader(trainset,
                batch_size=conf['batch_size'], shuffle=True, num_workers=4)
    loaders['test'] = torch.utils.data.DataLoader(testset,
                batch_size=conf['batch_size'], shuffle=False, num_workers=4)
    print(9*'#', 'Using {} dataset'.format(conf['dataset']), 9*'#')


    # Training
    use_gpu  = not args.disable_gpu and torch.cuda.is_available()
    caps_net = CapsNetTrainer(loaders,
                              conf['model'],
                              conf['lr'],
                              conf['lr_decay'],
                              conf['num_classes'],
                              conf['num_routing'],
                              conf['loss'],
                              use_gpu=use_gpu,
                              multi_gpu=args.multi_gpu)

    ensure_dir('logs') #
    logger = {}
    logger['train'] = Logger('logs/{}-train'.format(conf['dataset']))
    logger['test'] = Logger('logs/{}-test'.format(conf['dataset']))
    ensure_dir(conf['save_dir']) #
    caps_net.train(conf['epochs'], classes, conf['save_dir'], logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Capsules Networks')
    parser.add_argument('-c', '--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    # Use multiple GPUs? '--multi_gpu' will store multi_gpu as True
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Flag whether to use multiple GPUs.')
    # Select GPU device
    parser.add_argument('--disable_gpu', action='store_true',
                help='Flag whether to use disable GPU')

    args = parser.parse_args()

    main(args)
