'''

References:
* https://github.com/enakai00/colab_jaxbook/blob/main/Chapter01/ \
    3.%20Least%20squares%20method%20with%20Flax%20and%20Optax.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt

from jax import random
from jax import numpy as jnp
from flax import linen as nn
from flax.training import train_state
import jax
import optax


def get_data():
    train_t = jnp.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4,
                         25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
    train_t = train_t.reshape([12, 1])

    train_x = jnp.asarray([[month**n for n in range(1, 5)]
                          for month in range(1, 13)])
    return train_t, train_x


class TemperatureModel(nn.Module):

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(features=1)(x)
        return y


@jax.jit
def loss_fn(params, state, inputs, labels):
    predicts = state.apply_fn({'params': params}, inputs)
    loss = optax.l2_loss(predicts, labels).mean()
    return loss


@jax.jit
def train_step(state, inputs, labels):
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, state, inputs, labels)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def plot_history(targets, x_range=None, y_range=None):
    fig, ax = plt.subplots()
    ax.plot(targets)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return ax


def main():

    train_t, train_x = get_data()
    print(train_t.shape)
    print(train_x.shape)

    key, key1 = random.split(random.PRNGKey(0))
    variables = TemperatureModel().init(key1, train_x)
    print(variables)

    state = train_state.TrainState.create(
        apply_fn=TemperatureModel().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001)
    )

    loss_history = []
    for step in range(1, 100001):
        state, loss_val = train_step(state, train_x, train_t)
        loss_history.append(jax.device_get(loss_val).tolist())
        if step % 10000 == 0:
            print('Step {}, Loss {:0.4f}'.format(step, loss_val),
                  flush=True)

    plot_history(loss_history, x_range=(0, 100))
    plot_history(loss_history, y_range=(0, 8))
    plt.show()
    plt.close()

    xs = np.linspace(1, 12, 100)
    inputs = jnp.asarray([[month**n for n in range(1, 5)]
                         for month in xs])
    ys = state.apply_fn({'params': state.params}, inputs)
    fig = plt.figure(figsize=(6, 4))
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xlim(1, 12)
    subplot.set_ylim(0, 30)
    subplot.set_xticks(range(1, 13))
    subplot.set_xlabel('Month')
    subplot.set_ylabel('℃')

    subplot.scatter(range(1, 13), train_t)
    _ = subplot.plot(xs, ys)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
