'''
'''
import matplotlib.pyplot as plt
import numpy as np

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax import random
from flax.training import train_state

from tools import plotting


class DoubleLayerModel(nn.Module):
    @nn.compact
    def __call__(self, x, get_logits=False):
        x = nn.Dense(features=2)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=2)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        if get_logits:
            return x
        x = nn.sigmoid(x)
        return x


def main():

    def generate_datablock(key, n, mu, cov, t):
        data = random.multivariate_normal(
            key, jnp.asarray(mu), jnp.asarray(cov), jnp.asarray([n]))
        data = jnp.hstack([data, jnp.ones([n, 1])*t])
        return data

    key, key1, key2, key3, key4, key5 = random.split(random.PRNGKey(0), 6)
    data1 = generate_datablock(key1, 24, [7,   7], [[18, 0], [0, 18]], 1)
    data2 = generate_datablock(key2, 24, [-7, -7], [[18, 0], [0, 18]], 1)
    data3 = generate_datablock(key3, 24, [7,  -7], [[18, 0], [0, 18]], 0)
    data4 = generate_datablock(key4, 24, [-7,  7], [[18, 0], [0, 18]], 0)

    data = random.permutation(key5, jnp.vstack([data1, data2, data3, data4]))
    train_x, train_t = jnp.split(data, [2], axis=-1)

    key, key1 = random.split(key)
    variables = DoubleLayerModel().init(key1, train_x)

    print(jax.tree_util.tree_map(lambda x: x.shape, variables['params']))

    state = train_state.TrainState.create(
        apply_fn=DoubleLayerModel().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001)
    )

    @jax.jit
    def loss_fn(params, state, inputs, labels):
        logits = state.apply_fn({'params': params}, inputs, get_logits=True)
        loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
        acc = jnp.mean(jnp.sign(logits) == jnp.sign(labels-0.5))
        return loss, acc

    @jax.jit
    def train_step(state, inputs, labels):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, inputs, labels
        )
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, acc

    loss_history, acc_history = [], []
    for step in range(1, 5001):
        state, loss, acc = train_step(state, train_x, train_t)
        loss_history.append(jax.device_get(loss).tolist())
        acc_history.append(jax.device_get(acc).tolist())
        if step % 1000 == 0:
            print('Step: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                step, loss, acc), flush=True)

    plotting.plot_history(loss_history)
    # plotting.eye()

    plotting.plot_history(acc_history)
    # plotting.eye()

    train_set0 = [jax.device_get(x).tolist()
                  for x, t in zip(train_x, train_t) if t == 0]
    train_set1 = [jax.device_get(x).tolist()
                  for x, t in zip(train_x, train_t) if t == 1]

    fig = plt.figure(figsize=(7, 7))
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_ylim([-15, 15])
    subplot.set_xlim([-15, 15])
    subplot.set_xlabel('x1')
    subplot.set_ylabel('x2')
    subplot.scatter([x for x, y in train_set1],
                    [y for x, y in train_set1], marker='x')
    subplot.scatter([x for x, y in train_set0],
                    [y for x, y in train_set0], marker='o')

    locations = [[x1, x2]
                 for x2 in np.linspace(-15, 15, 500)
                 for x1 in np.linspace(-15, 15, 500)]
    p_vals = state.apply_fn({'params': state.params},
                            np.array(locations)).reshape([500, 500])
    _ = subplot.imshow(p_vals, origin='lower', extent=(-15, 15, -15, 15),
                       vmin=0, vmax=1, cmap=plt.cm.gray_r, alpha=0.4)
    plotting.eye()


if __name__ == "__main__":
    main()
