'''
'''
import numpy as np
import matplotlib.pyplot as plt

import jax
import optax
from jax import random
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state

from tools import plotting


def generate_tarining_data(keys):

    key1, key2, key3 = keys

    n0 = 20
    mu0 = [10, 11]
    varince0 = 20
    data0 = random.multivariate_normal(
        key1, jnp.asarray(mu0), jnp.eye(2)*varince0, jnp.asarray([n0])
    )
    data0 = jnp.hstack([data0, jnp.zeros([n0, 1])])

    n1 = 15
    mu1 = [18, 20]
    varince1 = 22
    data1 = random.multivariate_normal(
        key2, jnp.asarray(mu1), jnp.eye(2)*varince1, jnp.asarray([n1])
    )
    data1 = jnp.hstack([data1, jnp.ones([n1, 1])])

    data = random.permutation(key3, jnp.vstack([data0, data1]))
    train_x, train_t = jnp.split(data, [2], axis=1)

    return train_x, train_t


class LogisticRegressionModel(nn.Module):

    @nn.compact
    def __call__(self, x, get_logits=False):
        x = nn.Dense(features=1)(x)
        if get_logits:
            return x
        x = nn.sigmoid(x)
        return x


@jax.jit
def loss_fn(params, state, inputs, labels):
    logits = state.apply_fn({'params': params}, inputs, get_logits=True)
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = jnp.mean(jnp.sign(logits) == jnp.sign(labels-0.5))
    return loss, acc


@jax.jit
def train_step(state, inputs, labels):
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state, inputs, labels)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc


def main():

    key, key1, key2, key3 = random.split(random.PRNGKey(0), 4)

    train_x, train_t = generate_tarining_data([key1, key2, key3])
    print(train_x[:10])

    key, key1 = random.split(key)
    variables = LogisticRegressionModel().init(key1, train_x)
    print(variables)

    state = train_state.TrainState.create(
        apply_fn=LogisticRegressionModel().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001)
    )
    print(state)

    loss_history = []
    acc_history = []
    for step in range(1, 10001):
        state, loss, acc = train_step(state, train_x, train_t)
        loss_history.append(jax.device_get(loss).tolist())
        acc_history.append(jax.device_get(acc).tolist())
        if step % 1000 == 0:
            print('Step {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                step, loss, acc), flush=True)

    plotting.plot_history(loss_history)
    plt.close()

    plotting.plot_history(acc_history)
    # plt.show()
    plt.close()

    [w1], [w2] = state.params['Dense_0']['kernel']
    [b] = state.params['Dense_0']['bias']

    train_set0 = [jax.device_get(x).tolist()
                  for x, t in zip(train_x, train_t) if t == 0]
    train_set1 = [jax.device_get(x).tolist()
                  for x, t in zip(train_x, train_t) if t == 1]

    fig = plt.figure(figsize=(7, 7))
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xlim([0, 30])
    subplot.set_ylim([0, 30])
    subplot.set_xlabel('x1')
    subplot.set_ylabel('x2')
    subplot.scatter([x for x, y in train_set1],
                    [y for x, y in train_set1], marker='x')
    subplot.scatter([x for x, y in train_set0],
                    [y for x, y in train_set0], marker='o')

    xs = np.linspace(0, 30, 10)
    ys = - (w1*xs/w2 + b/w2)
    subplot.plot(xs, ys)

    locations = [[x1, x2]
                 for x2 in np.linspace(0, 30, 100)
                 for x1 in np.linspace(0, 30, 100)]
    p_vals = state.apply_fn(
        {'params': state.params}, np.array(locations)).reshape([100, 100])
    _ = subplot.imshow(p_vals, origin='lower', extent=(0, 30, 0, 30),
                       vmin=0, vmax=1, cmap=plt.cm.gray_r, alpha=0.4)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
