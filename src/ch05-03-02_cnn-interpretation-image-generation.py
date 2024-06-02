'''
'''
import pathlib

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
from jax import random
from flax.training import checkpoints
from flax.training import train_state

from models import DoubleLayerCNN


def main():

    class TrainState(train_state.TrainState):
        epoch: int
        dropout_rng: type(random.PRNGKey(0))

    variables = DoubleLayerCNN().init(
        random.PRNGKey(0), jnp.zeros([1, 28, 28])
    )
    state = TrainState.create(
        apply_fn=DoubleLayerCNN().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001),
        dropout_rng=random.PRNGKey(0),
        epoch=0
    )

    state = checkpoints.restore_checkpoint(
        ckpt_dir=pathlib.Path('../models/').resolve(),
        prefix='DoubleLayerCNN_checkpoint_',
        target=state
    )

    @jax.jit
    def first_filter_output_mean(image, i):
        filter_output = state.apply_fn(
            {'params': state.params},
            jnp.asarray([image]),
            get_filter_output1=True
        )
        return jnp.mean(filter_output[0, :, :, i])

    @jax.jit
    def second_filter_output_mean(image, i):
        filter_output = state.apply_fn(
            {'params': state.params},
            jnp.asarray([image]),
            get_filter_output2=True
        )
        return jnp.mean(filter_output[0, :, :, i])

    def create_pattern(filter_output_mean, i):
        key = random.fold_in(random.PRNGKey(0), i)
        image = random.normal(key, (28, 28)) * 0.1 + 0.5

        epsilon = 1000
        for _ in range(50):
            image += epsilon * jax.grad(filter_output_mean)(image, i)
            epsilon *= 0.9

        image -= jnp.min(image)
        image /= jnp.max(image)

        return jax.device_get(image)

    fig = plt.figure(figsize=(16, 2))
    for i in range(32):
        img = create_pattern(first_filter_output_mean, i)
        subplot = fig.add_subplot(2, 16, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(img.reshape((28, 28)), vmin=0, vmax=1,
                       cmap=plt.cm.gray_r)
    # plt.show()
    plt.close()

    fig = plt.figure(figsize=(16, 4))
    for i in range(64):
        img = create_pattern(second_filter_output_mean, i)
        subplot = fig.add_subplot(4, 16, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(img.reshape((28, 28)), vmin=0, vmax=1,
                       cmap=plt.cm.gray_r)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
