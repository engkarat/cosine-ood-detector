import numpy as np
import os
import pickle
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_cifar100():
    tr_data = unpickle(os.path.join(project_path, 'data/cifar-100/train'))
    test_data = unpickle(os.path.join(project_path, 'data/cifar-100/test'))

    x_train = tr_data['data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])/255.
    x_test = test_data['data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])/255.
    y_train = np.array(tr_data['fine_labels'])
    y_test = np.array(test_data['fine_labels'])
    return x_train, y_train, x_test, y_test
