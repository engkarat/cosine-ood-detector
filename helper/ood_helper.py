import cv2
import numpy as np
import os
import sklearn.metrics as sk
import sys
from scipy.io import loadmat

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from data.load_cifar10 import load_cifar10
from data.load_cifar100 import load_cifar100


def std_mean_dataset_per_channel(image_batch, normalize_mean, normalize_std):
    imgs = np.zeros_like(image_batch)
    for i in range(3):
        imgs[:, :, :, i] = ( image_batch[:, :, :, i] - normalize_mean[i] ) / normalize_std[i]
    return imgs


def read_img_cv2(img_path, normalizing_term=1):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im/float(normalizing_term)


def read_imgs_folder(im_dir):
    f_names = os.listdir(im_dir)
    imgs = []
    for f_name in f_names:
        img = read_img_cv2(os.path.join(im_dir, f_name), normalizing_term=255.)
        imgs.append(img)
    return np.array(imgs)


def get_ood_dataset_cifar(mean, std):
    # LSUN cropped
    print('Loading LSUN (c)')
    lsun_cropped = read_imgs_folder('{}/data/ood_dataset/lsun_cropped_mod/test/'.format(project_path))
    lsun_cropped = lsun_cropped[:, 2: 34, 2: 34]
    lsun_cropped = std_mean_dataset_per_channel(lsun_cropped, mean, std)
    # LSUN resized
    print('Loading LSUN (r)')
    lsun_resized = read_imgs_folder('{}/data/ood_dataset/lsun_resized/test/'.format(project_path))
    lsun_resized = std_mean_dataset_per_channel(lsun_resized, mean, std)
    # Tiny ImageNet Cropped
    print('Loading TinyImNet (c)')
    imnet_cropped = read_imgs_folder('{}/data/ood_dataset/imnet_cropped_mod/test/'.format(project_path))
    imnet_cropped = imnet_cropped[:, 2: 34, 2: 34]
    imnet_cropped = std_mean_dataset_per_channel(imnet_cropped, mean, std)
    # Tiny ImageNet Resized
    print('Loading TinyImNet (r)')
    imnet_resized = read_imgs_folder('{}/data/ood_dataset/imnet_resized/test/'.format(project_path))
    imnet_resized = std_mean_dataset_per_channel(imnet_resized, mean, std)
    # iSUN
    print('Loading iSUN')
    isun = read_imgs_folder('{}/data/ood_dataset/isun/test/'.format(project_path))
    isun = std_mean_dataset_per_channel(isun, mean, std)
    # SVHN
    print('Loading SVHN')
    # svhn = loadmat('{}/data/ood_dataset/svhn/test_32x32.mat'.format(project_path))
    # svhn = svhn['X'].transpose([3, 0, 1, 2]) / 255.
    # np.random.seed(555)
    # svhn = (svhn - mean) / std
    svhn = read_imgs_folder('{}/data/ood_dataset/svhn/test/'.format(project_path))
    svhn = std_mean_dataset_per_channel(svhn, mean, std)
    # Gaussian Noise
    np.random.seed(555)
    print('Loading Gaussian Noise')
    gaus_noise = np.random.normal(loc=0.5, size=[10000, 32, 32, 3])
    gaus_noise[gaus_noise > 1] = 1
    gaus_noise[gaus_noise < 0] = 0
    gaus_noise = std_mean_dataset_per_channel(gaus_noise, mean, std)
    # # Uniform Noise
    print('Loading Uniform Noise')
    unif_noise = np.random.uniform(size=[10000, 32, 32, 3])
    unif_noise = std_mean_dataset_per_channel(unif_noise, mean, std)

    datasets = {}
    datasets['lsun_cropped'] = lsun_cropped
    datasets['lsun_resized'] = lsun_resized
    datasets['imnet_cropped'] = imnet_cropped
    datasets['imnet_resized'] = imnet_resized
    datasets['isun'] = isun
    datasets['svhn'] = svhn
    datasets['gaus_noise'] = gaus_noise
    datasets['unif_noise'] = unif_noise
    return datasets
