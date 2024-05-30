'''

Topics
'''
import pickle

import matplotlib.pyplot as plt
import numpy as np

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jax import random

from tools import plotting


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
    return fig


def show_hidden_vals(subplot, hidden_vals, labels):
    z1_vals = [[], [], []]
    z2_vals = [[], [], []]

    for hidden_val, label in zip(hidden_vals, labels):
        label_num = np.argmax(label)
        z1_vals[label_num].append(hidden_val[0])
        z2_vals[label_num].append(hidden_val[1])

    subplot.set_xlim([-1.1, 1.1])
    subplot.set_ylim([-1.1, 1.1])
    subplot.set_aspect('equal')
    a = subplot.scatter(z1_vals[0], z2_vals[0],
                        s=100, marker='|', color='blue')
    b = subplot.scatter(z1_vals[1], z2_vals[1],
                        s=100, marker='_', color='orange')
    c = subplot.scatter(z1_vals[2], z2_vals[2],
                        s=100, marker='+', color='green')
    return [a, b, c]


class DynamicFilterClassificationModel(nn.Module):
    @nn.compact
    def __call__(self, x, get_logits=False,
                 get_filter_output=False, get_pooling_output=False):
        x = x.reshape([-1, 28, 28, 1])
        x = nn.Conv(features=2, kernel_size=(5, 5), use_bias=False,
                    name='ConvLayer')(x)
        x = jnp.abs(x)
        if get_filter_output:
            return x
        x = nn.max_pool(x, window_shape=(28, 28), strides=(28, 28))
        if get_pooling_output:
            return x
        x = x.reshape([x.shape[0], -1])
        x = nn.Dense(features=3)(x)
        if get_logits:
            return x
        x = nn.softmax(x)
        return x


def main():
    with open('../data/ORENIST.pkl', 'rb') as file:
        images, labels = pickle.load(file)
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)
    # plot_images(images, labels)

    key, key1 = random.split(random.PRNGKey(0))
    variables = DynamicFilterClassificationModel().init(key1, images[0:1])
    print(jax.tree_util.tree_map(lambda x: x.shape, variables['params']))

    state = train_state.TrainState.create(
        apply_fn=DynamicFilterClassificationModel().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001)
    )

    @jax.jit
    def loss_fn(params, state, inputs, labels):
        logits = state.apply_fn({'params': params}, inputs, get_logits=True)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
        return loss, acc

    @jax.jit
    def train_step(state, inputs, labels):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, inputs, labels)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, acc

    loss_history, acc_history = [], []
    for step in range(1, 2001):
        state, loss, acc = train_step(state, images, labels)
        loss_history.append(jax.device_get(loss).tolist())
        acc_history.append(jax.device_get(acc).tolist())
        if step % 20 == 0:
            print('Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                step, loss, acc), flush=True)

    plotting.plot_history(acc_history, "Accuracy")
    # plt.show()
    plt.close()
    plotting.plot_history(loss_history, "Loss")
    # plt.show()
    plt.close()

    filter_vals = jax.device_get(state.params['ConvLayer']['kernel'])
    conv_output = jax.device_get(
        state.apply_fn({'params': state.params}, images[:9],
                       get_filter_output=True)
    )
    plot_filter_filtered_images(images, labels, filter_vals, conv_output,
                                original_image_size=[28, 28])
    # plt.show()
    plt.close()

    pool_output = jax.device_get(
        state.apply_fn({'params': state.params}, images[:9],
                       get_pooling_output=True)
    )
    plot_filter_filtered_images(images, labels, filter_vals, pool_output,
                                original_image_size=[28, 28])
    # plt.show()
    plt.close()

    bin_index = np.sign(pool_output-8.0)
    plot_filter_filtered_images(images, labels, filter_vals, bin_index,
                                original_image_size=[28, 28])
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
