from __future__ import division
from torchvision import transforms, utils
import argparse
import math
import numpy as np
import os
import socket
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)
from data.cifar_loader import Cifar10Dataset, Cifar100Dataset
from data.load_cifar10 import load_cifar10
from data.load_cifar100 import load_cifar100
from nets.nets_cosine import DenseNetCosine, WideResNetCosine
from torch_model.model import Model, SGDNoWeightDecayLast
import helper.common_helper as com_help
import helper.ood_helper as ood_help


parser = argparse.ArgumentParser(
    description='Out-of-distribution detection, neural network training.'
)
parser.add_argument(
    '--nn', default="wrn-28-10", type=str,
    help='neural network name'
)
parser.add_argument(
    '--tr_dset', default="cifar10", type=str,
    help='training (in-distribution) dataset'
)


dataset_details = {
    'cifar10': {
        'n_class': 10,
        'mean': [0.49137255, 0.48235294, 0.44666667],
        'std': [0.24705882, 0.24352941, 0.26156863],
        'dset': Cifar10Dataset,
        'n_channel': 3,
    },
    'cifar100': {
        'n_class': 100,
        'mean': [0.5071, 0.4865, 0.4409],
        'std': [0.2673, 0.2564, 0.2762],
        'dset': Cifar100Dataset,
        'n_channel': 3,
    },
}


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.nn is not None, "Please specify '--nn'."
    assert args.tr_dset is not None, "Please specify '--tr_dset'."

    # General details
    gpu_amount = 1
    tr_dset_name = args.tr_dset
    network_name = args.nn
    gpus = range(0, gpu_amount)
    ckpt_path = os.path.join(project_path, 'ckpt', 'cos', network_name, tr_dset_name)
    print(args)
    print("Running on: {}".format(socket.gethostname()))

    # Dataset details
    n_class = dataset_details[tr_dset_name]['n_class']
    mean = dataset_details[tr_dset_name]['mean']
    std = dataset_details[tr_dset_name]['std']
    Dataset = dataset_details[tr_dset_name]['dset']
    n_ch = dataset_details[tr_dset_name]['n_channel']

    # Transforms applied to training image
    composed = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    tr_dset = Dataset('train', composed)
    tr_batch_size = 64 if 'dense' in network_name else 128
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=tr_batch_size, shuffle=True, num_workers=4)

    # Transforms applied to testing image
    composed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    te_dset = Dataset('test', composed)
    te_loader = torch.utils.data.DataLoader(
        te_dset, batch_size=128, shuffle=False, num_workers=4)

    if 'wrn' in network_name:
        _, n_layer, widen = network_name.split('-')
        net = WideResNetCosine(
            int(n_layer), n_class, widen_factor=int(widen), input_n_channel=n_ch,
        )
        w_decay = 5e-4
        epoch = 200
        saved_epoch = [200,]
    elif 'dense' in network_name:
        _, n_layer = network_name.split('-')
        net = DenseNetCosine(
            int(n_layer), n_class, input_n_channel=n_ch,
        )
        w_decay = 1e-4
        epoch = 300
        saved_epoch = [300,]

    params = net.parameters()
    criterion = nn.CrossEntropyLoss()
    optim_fn = SGDNoWeightDecayLast
    optimizer = optim_fn(
        params, lr=1e-3, momentum=0.9, nesterov=True, weight_decay=w_decay,
    )

    def lr_mul(step, all_epoch):
        epoch = step // len(tr_loader)
        if epoch < (all_epoch * 0.5):
            return 100
        elif epoch < (all_epoch * 0.75):
            return 10
        elif epoch <= (all_epoch * 1.0):
            return 1
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_mul(step, epoch)
    )

    model = Model(gpus, ckpt_path, net, optimizer, criterion, scheduler)
    model.fit(
        epoch, tr_loader, te_loader, save_at=saved_epoch
    )
    val_loss, val_acc = model.validate(te_loader)
    print("Model evaluation with testing dataset: {:.2f}%".format(val_acc))

    # OOD testing dataset
    _, _, cifar10_x, cifar10_y = load_cifar10()
    cifar10_x = (cifar10_x - mean) / std
    _, _, cifar100_x, cifar100_y = load_cifar100()
    cifar100_x = (cifar100_x - mean) / std
    datasets = ood_help.get_ood_dataset_cifar(mean, std)
    datasets['cifar10'] = cifar10_x
    datasets['cifar100'] = cifar100_x

    # OOD Evaluation
    keys_preds = ['scaled_cosine', 'softmax', 'scale', 'cosine_similarity']
    preds = com_help.get_predictions(model, datasets, keys_preds)
    
    if tr_dset_name in ['cifar10']:
        lbl = cifar10_y
    elif tr_dset_name in ['cifar100']:
        lbl = cifar100_y

    ood_datasets = ['cifar10'] if tr_dset_name == 'cifar100' else ['cifar100']
    ood_datasets += [
        'imnet_cropped', 'imnet_resized', 'lsun_cropped', 'lsun_resized',
        'isun', 'svhn', 'gaus_noise', 'unif_noise',
    ]
    com_help.ood_detection_eval(
        preds, tr_dset_name, 'cosine_similarity', ood_datasets
    )
