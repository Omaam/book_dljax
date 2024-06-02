'''
'''
import pathlib
from functools import partial
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import optax
from jax import random
from flax.training import train_state
from flax.training import checkpoints

from models.models import DoubleLayerCNN
from tools.data import create_batches
from tools.data import get_data_mnist
from tools.plotting import plot_history


class TrainState(train_state.TrainState):
    epoch: int
    dropout_rng: type(random.PRNGKey(0))


def main():

    (
     (train_images, train_labels),
     (test_images, test_labels)
    ) = get_data_mnist('../data/mnist.npz')

    key, key1 = random.split(random.PRNGKey(0))
    variables = DoubleLayerCNN().init(
        key1, train_images[0:1], {'dropout': random.PRNGKey(0)}
    )

    pprint(jax.tree_util.tree_map(lambda x: x.shape, variables['params']))

    key, key1 = random.split(key)
    state = TrainState.create(
        apply_fn=DoubleLayerCNN().apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=0.001),
        dropout_rng=key1,
        epoch=0
    )

    checkpoints.save_checkpoint(
        ckpt_dir=pathlib.Path('../models/').resolve(),
        prefix='DoubleLayerCNN_checkpoint_',
        target=state,
        step=state.epoch,
        overwrite=True
    )

    @partial(jax.jit, static_argnames=['eval'])
    def loss_fn(params, state, inputs, labels, dropout_rng, eval):
        logits = state.apply_fn(
            {'params': params}, inputs, get_logits=True, eval=eval,
            rngs={'dropout': dropout_rng}
        )
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        acc = jnp.mean(
            jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)
        )
        return loss, acc

    @partial(jax.jit, static_argnames=['eval'])
    def train_step(state, inputs, labels, eval):
        if not eval:
            new_dropout_rng, dropout_rng = random.split(state.dropout_rng)
            (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, state, inputs, labels, dropout_rng, eval
            )
            new_state = state.apply_gradients(
                grads=grads, dropout_rng=new_dropout_rng
            )
        else:
            loss, acc = loss_fn(
                state.params, state, inputs, labels, random.PRNGKey(0), eval
            )
            new_state = state

        return new_state, loss, acc,

    def train_epoch(state, input_batched, label_batched, eval):
        loss_history, acc_history = [], []
        for inputs, labels in zip(input_batched, label_batched):
            new_state, loss, acc = train_step(state, inputs, labels, eval)
            if not eval:
                state = new_state
            loss_history.append(jax.device_get(loss).tolist())
            acc_history.append(jax.device_get(acc).tolist())
        return state, np.mean(loss_history), np.mean(acc_history)

    def fit(state, ckpt_dir, prefix,
            train_inputs, train_labels, test_inputs, test_labels,
            epochs, batch_size):

        state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir, prefix=prefix, target=state
        )
        train_inputs_batched = create_batches(train_inputs, batch_size)
        train_labels_batched = create_batches(train_labels, batch_size)
        test_inputs_batched = create_batches(test_inputs, batch_size)
        test_labels_batched = create_batches(test_labels, batch_size)

        loss_history_train, acc_history_train = [], []
        loss_history_test, acc_history_test = [], []

        for epoch in range(state.epoch + 1, state.epoch + 1 + epochs):

            # Training
            state, loss_train, acc_train = train_epoch(
                state, train_inputs_batched, train_labels_batched, eval=False)
            loss_history_train.append(loss_train)
            acc_history_train.append(acc_train)

            # Evaluation
            _, loss_test, acc_test = train_epoch(
                state, test_inputs_batched, test_labels_batched, eval=True)
            loss_history_test.append(loss_test)
            acc_history_test.append(acc_test)

            print('Epoch: {} | '.format(epoch), end='', flush=True)
            print('Loss: {:.4f}, Accuracy: {:.4f} | '.format(
                loss_train, acc_train), end='', flush=True)
            print('Loss: {:.4f}, Accuracy: {:.4f}'.format(
                loss_test, acc_test), flush=True)

            state = state.replace(epoch=state.epoch+1)
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir, prefix=prefix,
                target=state, step=state.epoch, overwrite=True, keep=5
            )

        history = {
            'loss_train': loss_history_train,
            'acc_train': acc_history_train,
            'loss_test': loss_history_test,
            'acc_test': acc_history_test,
        }

        return state, history

    ckpt_dir = pathlib.Path('../models/').resolve()
    prefix = 'DoubleLayerCNN_checkpoint_'
    state, history = fit(state, ckpt_dir, prefix,
                         train_images, train_labels, test_images, test_labels,
                         epochs=16, batch_size=128)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0] = plot_history(history['acc_train'], ax=ax[0], label='acc_train')
    ax[0] = plot_history(history['acc_test'], ax=ax[0], label='acc_test')
    ax[1] = plot_history(history['loss_train'], ax=ax[1], label='loss_train')
    ax[1] = plot_history(history['loss_test'], ax=ax[1], label='loss_test')
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
