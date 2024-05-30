'''単層畳み込みニューラルネットによる手書き文字分類の結果確認。

トピック
* 畳み込みフィルター後の画像の確認。
* プーリング層を通過したあとの画像の確認。
* 誤推定の手書き文字の目視による確認。
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import jax
import optax
from jax import random
from flax.training import train_state
from flax.training import checkpoints

from tools import loading
from models.models import SingleLayerCNN


def plot_filter_filtered_images(images, labels, filter_vals, filtered_images,
                                num_filters):
    fig = plt.figure(figsize=(10, num_filters+1))
    v_max = np.max(filtered_images)

    for i in range(num_filters):
        subplot = fig.add_subplot(num_filters+1, 10, 10*(i+1)+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(filter_vals[:, :, 0, i], cmap=plt.cm.gray_r)

    for i in range(9):
        subplot = fig.add_subplot(num_filters+1, 10, i+2)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title(np.argmax(labels[i]))
        subplot.imshow(images[i].reshape([28, 28]),
                       vmin=0, vmax=1, cmap=plt.cm.gray_r)

        for f in range(num_filters):
            subplot = fig.add_subplot(num_filters+1, 10, 10*(f+1)+i+2)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.imshow(filtered_images[i, :, :, f],
                           vmin=0, vmax=v_max, cmap=plt.cm.gray_r)


def main():
    (
     (train_images, train_labels),
     (test_images, test_labels)
    ) = loading.get_data_mnist()

    variables = SingleLayerCNN().init(random.PRNGKey(0), test_images[0:1])

    state = train_state.TrainState.create(
        apply_fn=SingleLayerCNN().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001)
    )
    print(jax.tree_util.tree_map(lambda x: x.shape, state.params))

    state = checkpoints.restore_checkpoint(
        ckpt_dir=pathlib.Path('../model').resolve(),
        prefix='SingleLayerCNN_checkpoint_',
        target=state
    )

    filter_vals = jax.device_get(state.params['ConvLayer']['kernel'])
    filter_output = jax.device_get(
        state.apply_fn({'params': state.params}, test_images[:9],
                       get_filter_output=True)
    )
    pooling_output = jax.device_get(
        state.apply_fn({'params': state.params}, test_images[:9],
                       get_pooling_output=True)
    )

    plot_filter_filtered_images(test_images, test_labels,
                                filter_vals, filter_output,
                                num_filters=16)
    # plt.show()
    plt.close()

    plot_filter_filtered_images(test_images, test_labels,
                                filter_vals, pooling_output,
                                num_filters=16)
    # plt.show()
    plt.close()

    predictions = jax.device_get(
        state.apply_fn({'params': state.params}, test_images))

    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(wspace=0.1, hspace=0.6)

    c = 0
    for i in range(len(predictions)):
        prediction = np.argmax(predictions[i])
        actual = np.argmax(test_labels[i])
        if prediction == actual:
            continue

        image = jax.device_get(test_images[i])
        prediction_vals = predictions[i]

        subplot = fig.add_subplot(5, 4, c*2+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('{} / {}'.format(prediction, actual))
        subplot.imshow(image.reshape([28, 28]),
                       vmin=0, vmax=1, cmap=plt.cm.gray_r)
        subplot = fig.add_subplot(5, 4, c*2+2)
        subplot.set_xticks(range(10))
        subplot.set_xlim([-0.5, 9.5])
        subplot.set_ylim([0, 1])
        subplot.bar(range(10), prediction_vals, align='center', edgecolor='b')
        c += 1
        if c == 10:
            break
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
