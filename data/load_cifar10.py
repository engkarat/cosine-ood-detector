import numpy as np
import os
import pickle
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


data_dir = os.path.join(project_path, 'data', 'cifar-10-batches-py')
tr_file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_file_names = ['test_batch', ]


def read_img(file_path):
    data = pickle.load(open(file_path, 'rb'))
    x = np.dstack((data[b'data'][:, :1024], data[b'data'][:, 1024:2048], data[b'data'][:, 2048:])) / 255.
    x = x.reshape([-1, 32, 32, 3])
    y = data[b'labels']
    return x, y


def load_data(mode):
    assert mode in ['train', 'test'], "specified mode is not defined"
    file_names = {'train': tr_file_names, 'test': test_file_names}
    x_all = []
    y_all = []
    for f_name in file_names[mode]:
        x, y = read_img(os.path.join(data_dir, f_name))
        x_np, y_np = np.array(x), np.array(y, dtype='uint8')
        x_all.append(x_np)
        y_all.append(y_np)
    x_all_np = np.concatenate(x_all)
    y_all_np = np.concatenate(y_all)
    return x_all_np, y_all_np


def load_cifar10():
    x_train, y_train = load_data('train')
    x_test, y_test = load_data('test')
    return x_train, y_train, x_test, y_test
