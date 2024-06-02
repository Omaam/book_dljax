'''
References
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

import jax
import optax
from jax import numpy as jnp
from jax import random
from flax.training import train_state
from flax.training import checkpoints

from tools import plotting
from models.models import SingleLayerCNN


def get_data_mnist():
    (
       (train_images, train_labels),
       (test_images, test_labels)
    ) = mnist.load_data()

    train_images = train_images.reshape([-1, 784]).astype('float32') / 255
    test_images = test_images.reshape([-1, 784]).astype('float32') / 255

    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    return (train_images, train_labels), (test_images, test_labels)


def create_batches(data, batch_size):
    num_batches, mod = divmod(len(data), batch_size)
    data_batched = np.split(data[:num_batches * batch_size], num_batches)
    if mod:  # Last batch is smaller than batch_size
        data_batched.append(data[num_batches * batch_size:])
    data_batched = [jnp.asarray(x) for x in data_batched]
    return data_batched


def main():
    (
        (train_images, train_labels),
        (test_images, test_labels),
    ) = get_data_mnist()

    key, key1 = random.split(random.PRNGKey(0))
    variables = SingleLayerCNN().init(key1, train_images[0:1])
    print(jax.tree_util.tree_map(lambda x: x.shape, variables['params']))

    state = train_state.TrainState.create(
        apply_fn=SingleLayerCNN().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001)
    )

    @jax.jit
    def loss_fn(params, state, inputs, labels):
        logits = state.apply_fn({'params': params}, inputs, get_logits=True)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = jnp.mean(
            jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)
        )
        return loss, acc

    @jax.jit
    def train_step(state, inputs, labels):
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, inputs, labels)
        new_state = state.apply_gradients(grads=grads)
        return new_state, loss, acc

    def train_epoch(state, input_batched, label_batched, eval):
        loss_history = []
        acc_history = []
        for inputs, labels in zip(input_batched, label_batched):
            new_state, loss, acc = train_step(state, inputs, labels)
            if not eval:
                state = new_state
            loss_history.append(jax.device_get(loss))
            acc_history.append(jax.device_get(acc))
        return state, np.mean(loss_history), np.mean(acc_history)

    def fit(state, train_inputs, train_labels, test_inputs, test_labels,
            epochs, batch_size):

        train_inputs_batched = create_batches(train_inputs, batch_size)
        train_labels_batched = create_batches(train_labels, batch_size)
        test_inputs_batched = create_batches(test_inputs, batch_size)
        test_labels_batched = create_batches(test_labels, batch_size)

        loss_history_train, acc_history_train = [], []
        loss_history_test, acc_history_test = [], []

        for epoch in range(1, epochs+1):

            # Training
            state, loss_train, acc_train = train_epoch(
                state, train_inputs_batched, train_labels_batched, eval=False)
            loss_history_train.append(loss_train)
            acc_history_train.append(acc_train)

            # Evaluation
            state, loss_test, acc_test = train_epoch(
                state, test_inputs_batched, test_labels_batched, eval=True)
            loss_history_test.append(loss_test)
            acc_history_test.append(acc_test)

            print('Epoch: {} | '.format(epoch), end='', flush=True)
            print('Loss: {:.4f}, Accuracy: {:.4f} | '.format(
                loss_train, acc_train), end='', flush=True)
            print('Loss: {:.4f}, Accuracy: {:.4f}'.format(
                loss_test, acc_test), flush=True)

        history = {
            'loss_train': loss_history_train,
            'acc_train': acc_history_train,
            'loss_test': loss_history_test,
            'acc_test': acc_history_test,
        }

        return state, history

    state, history = fit(state,
                         train_images, train_labels, test_images, test_labels,
                         epochs=16, batch_size=128)

    checkpoints.save_checkpoint(
        ckpt_dir=pathlib.Path('../models').resolve(),
        prefix='SingleLayerCNN_checkpoint_',
        target=state, step=16, overwrite=True
    )

    ax = plotting.plot_history(history['acc_train'], label='Accuracy (Train)')
    plotting.plot_history(history['acc_test'], label='Accuracy (Test)', ax=ax)
    plt.show()
    plt.close()

    ax = plotting.plot_history(history['loss_train'], label='Loss (Train)')
    plotting.plot_history(history['loss_test'], label='Loss (Test)', ax=ax)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
