'''
'''
import pathlib

import numpy as np
from tensorflow.keras.datasets import mnist


def get_data_mnist():
    (
       (train_images, train_labels),
       (test_images, test_labels)
    ) = mnist.load_data(path=pathlib.Path('../data/mnist.npz').resolve())

    train_images = train_images.reshape([-1, 784]).astype('float32') / 255
    test_images = test_images.reshape([-1, 784]).astype('float32') / 255

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    return (train_images, train_labels), (test_images, test_labels)
