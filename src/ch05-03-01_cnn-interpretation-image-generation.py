'''
'''
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import optax
from jax import random
from flax.core import freeze
from flax.core import unfreeze
from flax.training import train_state

from models import FixedConvFilterModel


def main():

    filter0 = np.array(
        [[2, 1, 0, -1, -2],
         [3, 2, 0, -2, -3],
         [4, 3, 0, -3, -4],
         [3, 2, 0, -2, -3],
         [2, 1, 0, -1, -2]]
    ) / 23.0
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

    variables = FixedConvFilterModel().init(
        random.PRNGKey(0),
        jnp.zeros((1, 28*28))
    )
    params = unfreeze(variables['params'])
    params['Conv_0']['kernel'] = filter_array
    new_params = freeze(params)

    state = train_state.TrainState.create(
        apply_fn=FixedConvFilterModel().apply,
        params=new_params,
        tx=optax.adam(learning_rate=0.001)
    )

    @jax.jit
    def filter_output_mean(image, i):
        filter_output = state.apply_fn(
            {'params': state.params}, jnp.asarray([image])
        )
        return jnp.mean(filter_output[0, :, :, i])

    def create_pattern(i):
        key = random.fold_in(random.PRNGKey(0), i)
        image = random.normal(key, (28, 28)) * 0.1 + 0.5

        epsilon = 1000
        for _ in range(20):
            image += epsilon * jax.grad(filter_output_mean)(image, i)
            epsilon += 0.9

        image -= jnp.min(image)
        image /= jnp.max(image)

        return jax.device_get(image)

    fig = plt.figure(figsize=(30, 15))
    for i in range(2):
        img = create_pattern(i)
        subplot = fig.add_subplot(2, 16, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(img.reshape((28, 28)), vmin=0, vmax=1,
                       cmap=plt.cm.gray_r)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
