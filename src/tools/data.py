'''
'''
import pathlib
import numpy as np
import jax.numpy as jnp

from tensorflow.keras.datasets import mnist


def create_batches(data, batch_size):
    num_batches, mod = divmod(len(data), batch_size)
    data_batched = np.split(data[:num_batches * batch_size], num_batches)
    if mod:  # Last batch is smaller than batch_size
        data_batched.append(data[num_batches * batch_size:])
    data_batched = [jnp.asarray(x) for x in data_batched]
    return data_batched


def get_data_mnist(mnist_path):
    (
       (train_images, train_labels),
       (test_images, test_labels)
    ) = mnist.load_data(path=pathlib.Path(mnist_path).resolve())

    train_images = train_images.reshape([-1, 784]).astype('float32') / 255
    test_images = test_images.reshape([-1, 784]).astype('float32') / 255

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    return (train_images, train_labels), (test_images, test_labels)
