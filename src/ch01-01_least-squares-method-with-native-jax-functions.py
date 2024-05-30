'''
'''
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import random
from jax import numpy as jnp

from tools.time import stopwatch


def get_data():
    train_t = jnp.array([5.2, 5.7, 8.6, 14.9, 18.2, 20.4,
                         25.5, 26.4, 22.8, 17.5, 11.1, 6.6])
    train_t = train_t.reshape([12, 1])

    train_x = jnp.asarray([[month**n for n in range(0, 5)]
                           for month in range(1, 13)])
    return train_t, train_x


@jax.jit
def predict(w, x):
    y = jnp.matmul(x, w)
    return y


@jax.jit
def loss_fn(w, train_x, train_t):
    y = predict(w, train_x)
    loss = jnp.mean((y - train_t)**2)
    return loss


@stopwatch
def main():
    train_t, train_x = get_data()

    key, key1 = random.split(random.PRNGKey(0))
    w = random.normal(key1, [5, 1])

    grad_loss = jax.jit(jax.grad(loss_fn))

    learning_rate = 1e-8 * 1.4
    for step in range(1, 5000001):
        grads = grad_loss(w, train_x, train_t)
        w = w - learning_rate * grads
        if step % 500000 == 0:
            loss_val = loss_fn(w, train_x, train_t)
            print('Step {}, Loss {:0.4f}'.format(step, loss_val),
                  flush=True)
    print(w)

    xs = np.linspace(1, 12, 100)
    inputs = jnp.asarray([[month**n for n in range(0, 5)]
                          for month in xs])
    ys = predict(w, inputs)

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
