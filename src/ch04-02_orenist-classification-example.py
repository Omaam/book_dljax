'''固定した畳み込みフィルターを用いた画像分類。

Topics
* ORENIST の画像を２つの固定畳み込みフィルターによって特徴量抽出。
* 全結合層でそれらの特徴量から分類。
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
    plt.show()
    plt.close()


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


class StaticFilterClassificationModel(nn.Module):
    @nn.compact
    def __call__(self, x, get_logits=False, get_hidden_output=False):
        x = x.reshape([-1, 28, 28, 1])
        x = nn.Conv(features=2, kernel_size=(5, 5), use_bias=False,
                    name='StaticConv')(x)
        x = jnp.abs(x)
        x = nn.relu(x-0.2)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape([x.shape[0], -1])
        x = nn.Dense(features=2)(x)
        x = nn.tanh(x)
        if get_hidden_output:
            return x
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
    variables = StaticFilterClassificationModel().init(key1, images[0:1])
    print(jax.tree_util.tree_map(lambda x: x.shape, variables['params']))

    # Freeze convolution filter.
    params = unfreeze(variables['params'])
    params['StaticConv']['kernel'] = jnp.asarray(filter_array)
    new_params = freeze(params)

    # convolution filter を固定するためにマスクを準備する。
    params_mask = jax.tree_util.tree_map(lambda x: True, new_params)
    params_mask = unfreeze(params_mask)
    params_mask['StaticConv']['kernel'] = False
    params_mask = freeze(params_mask)

    # convolution filter 以外を更新する optimizer を定義。
    zero_grads = optax.GradientTransformation(
        # init_fn(_)
        lambda x: (),
        # update_fn(updates, state, params=None)
        lambda updates, state, params: (
            jax.tree.map(jnp.zeros_like, updates), ())
    )
    optimizer = optax.multi_transform(
        {True: optax.adam(learning_rate=0.001), False: zero_grads},
        params_mask
    )

    state = train_state.TrainState.create(
        apply_fn=StaticFilterClassificationModel().apply,
        params=new_params,
        tx=optimizer
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
    hidden_vals_history = []
    for step in range(1, 201):
        state, loss, acc = train_step(state, images, labels)
        loss_history.append(jax.device_get(loss).tolist())
        acc_history.append(jax.device_get(acc).tolist())

        hidden_vals = jax.device_get(
            state.apply_fn({'params': state.params}, images,
                           get_hidden_output=True)
        )
        hidden_vals_history.append(hidden_vals)

        if step % 20 == 0:
            print('Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                step, loss, acc), flush=True)

    plotting.plot_history(acc_history, "Accuracy")
    plotting.plot_history(loss_history, "Loss")
    plt.close()

    hidden_vals = state.apply_fn(
        {'params': state.params}, images, get_hidden_output=True)
    fig = plt.figure(figsize=(8, 8))
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xlabel('z1')
    subplot.set_ylabel('z2')
    _ = show_hidden_vals(subplot, hidden_vals, labels)
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(16, 8))
    for c, step in enumerate(range(0, 200, 25)):
        hidden_vals = hidden_vals_history[step]
        subplot = fig.add_subplot(2, 4, c+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('Step={}'.format(step+1))
        _ = show_hidden_vals(subplot, hidden_vals, labels)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
