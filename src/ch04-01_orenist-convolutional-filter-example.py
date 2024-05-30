'''畳み込みフィルターとプーリングの実例による理解。

Topics
* 畳み込みフィルターの最適化は行わず、画像の変換を理解する。
* プーリングによって画像の情報が落とされる様子を見る。
'''
import pickle

import matplotlib.pyplot as plt
import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.core.frozen_dict import freeze
from flax.core.frozen_dict import unfreeze
from jax import random


def plot_images(images, labels):
    fig = plt.figure(figsize=(10, 5))
    for i in range(40):
        subplot = fig.add_subplot(4, 10, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title(np.argmax(labels[i]))
        subplot.imshow(images[i].reshape([28, 28]),
                       vmin=0, vmax=1, cmap=plt.cm.gray_r)
    plt.show()
    plt.close()


def plot_filter_filtered_images(images, labels,
                                filter_vals, filtered_images,
                                original_image_size):
    fig = plt.figure(figsize=(10, 3))

    for i in range(2):
        subplot = fig.add_subplot(3, 10, 10*(i+1)+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(filter_vals[:, :, 0, i], cmap=plt.cm.gray_r)

    v_max = np.max(filtered_images)
    for i in range(9):
        subplot = fig.add_subplot(3, 10, i+2)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title(np.argmax(labels[i]))
        subplot.imshow(images[i].reshape(original_image_size),
                       vmin=0, vmax=1, cmap=plt.cm.gray_r)

        subplot = fig.add_subplot(3, 10, 10+i+2)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(filtered_images[i, :, :, 0],
                       vmin=0, vmax=v_max, cmap=plt.cm.gray_r)

        subplot = fig.add_subplot(3, 10, 20+i+2)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(filtered_images[i, :, :, 1],
                       vmin=0, vmax=v_max, cmap=plt.cm.gray_r)
    plt.show()
    plt.close()


class FixedConvFilterModel(nn.Module):
    @nn.compact
    def __call__(self, x, apply_pooling=False):
        x = x.reshape([-1, 28, 28, 1])
        x = nn.Conv(features=2, kernel_size=(5, 5), use_bias=False)(x)
        x = jnp.abs(x)
        x = nn.relu(x-0.2)
        if apply_pooling:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


def main():
    with open('../data/ORENIST.pkl', 'rb') as file:
        images, labels = pickle.load(file)
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)
    # plot_images(images, labels)

    filter0 = np.array(
        [[2, 1, 0, -1, -2],
         [3, 2, 0, -2, -3],
         [4, 3, 0, -3, -4],
         [3, 2, 0, -2, -3],
         [2, 1, 0, -1, -2]]
    ) / 23.
    filter1 = np.array(
        [[2,   3,  4,  3,  2],
         [1,   2,  3,  2,  1],
         [0,   0,  0,  0,  0],
         [-1, -2, -3, -2, -1],
         [-2, -3, -4, -3, -2]]
    ) / 23.0
    filter_array = np.zeros([5, 5, 1, 2])
    filter_array[..., 0, 0] = filter0
    filter_array[..., 0, 1] = filter1

    key, key1 = random.split(random.PRNGKey(0))
    variables = FixedConvFilterModel().init(key1, images[0:1])
    print(jax.tree_util.tree_map(lambda x: x.shape, variables['params']))

    params = unfreeze(variables['params'])
    params['Conv_0']['kernel'] = jnp.asarray(filter_array)
    new_params = freeze(params)

    state = train_state.TrainState.create(
        apply_fn=FixedConvFilterModel().apply,
        params=new_params,
        tx=optax.adam(learning_rate=0.001)
    )

    conv_output = jax.device_get(
        state.apply_fn({'params': state.params}, images[:9])
    )
    filter_vals = jax.device_get(state.params['Conv_0']['kernel'])
    plot_filter_filtered_images(images, labels, filter_vals, conv_output,
                                original_image_size=[28, 28])

    pool_output = jax.device_get(
        state.apply_fn({'params': state.params}, images[:9],
                       apply_pooling=True)
    )
    filter_vals = jax.device_get(state.params['Conv_0']['kernel'])
    plot_filter_filtered_images(images, labels, filter_vals, pool_output,
                                original_image_size=[28, 28])


if __name__ == '__main__':
    main()
