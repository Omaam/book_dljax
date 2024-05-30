'''
'''
import flax.linen as nn


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
