'''
'''
import flax.linen as nn
import jax.numpy as jnp


class FixedConvFilterModel(nn.Module):
    @nn.compact
    def __call__(self, x, apply_pooling=False):
        x = x.reshape([-1, 28, 28, 1])
        x = nn.Conv(features=2, kernel_size=(5, 5), use_bias=False)(x)
        x = jnp.abs(x)
        x = nn.relu(x-0.2)
        if apply_pooling:
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class SingleLayerCNN(nn.Module):

    @nn.compact
    def __call__(self, x, get_logits=False,
                 get_filter_output=False, get_pooling_output=False):
        x = x.reshape([-1, 28, 28, 1])
        x = nn.Conv(features=16, kernel_size=[5, 5], use_bias=True,
                    name='ConvLayer')(x)
        x = nn.relu(x)
        if get_filter_output:
            return x
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        if get_pooling_output:
            return x
        x = x.reshape([x.shape[0], -1])
        x = nn.Dense(features=1024)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        if get_logits:
            return x
        x = nn.softmax(x)
        return x


class DoubleLayerCNN(nn.Module):
    @nn.compact
    def __call__(self,
                 x,
                 get_logits=False,
                 eval=True,
                 get_filter_output1=False,
                 get_filter_output2=False):

        assert not (get_filter_output1 and get_filter_output2)

        x = x.reshape([-1, 28, 28, 1])
        batch_shape = x.shape[0]

        x = nn.Conv(features=32, kernel_size=(5, 5), use_bias=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        if get_filter_output1:
            return x

        x = nn.Conv(features=64, kernel_size=(5, 5), use_bias=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        if get_filter_output2:
            return x

        x = x.reshape([batch_shape, -1])
        x = nn.Dense(features=1024)(x)
        x = nn.relu(x)

        x = nn.Dropout(0.5, deterministic=eval)(x)

        x = nn.Dense(features=10)(x)
        if get_logits:
            return x
        x = nn.softmax(x)
        return x
