import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input, BatchNormalization
from keras.regularizers import l2

import net_blocks

class Encoder(object):
    pass

class Decoder(object):
    pass

class DenseEncoder(Encoder):
    def __init__(self, intermediate_dims, activation, wd, use_bn):
        self.intermediate_dims = intermediate_dims
        self.activation = activation
        self.wd = wd
        self.use_bn = use_bn

    def __call__(self, x):
        h = Flatten()(x)
        layers = net_blocks.dense_block(self.intermediate_dims, self.wd, self.use_bn, self.activation)
        for layer in layers:
            h = layer(h)
        return h

def decoder_layers(intermediate_dims, original_shape, activation, wd, use_bn):
    layers = []
    for intermediate_dim in reversed(intermediate_dims):
        layers.append(Dense(intermediate_dim, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization(mode=2))
        layers.append(Activation(activation))
    layers.append(Dense(np.prod(original_shape), activation='sigmoid', name="decoder_top", kernel_regularizer=l2(wd)))
    layers.append(Reshape(original_shape))
    return layers
    
class DenseDecoder(Decoder):
    def __init__(self, latent_dim, intermediate_dims, original_shape, activation, wd, use_bn):
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.original_shape = original_shape
        self.activation = activation
        self.wd = wd
        self.use_bn = use_bn

    def __call__(self, recons_input):
        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        layers = decoder_layers(self.intermediate_dims, self.original_shape, self.activation, self.wd, self.use_bn)

        generator_input = Input(shape=(self.latent_dim,))
        generator_output = generator_input
        recons_output = recons_input
        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)

        return generator_input, recons_output, generator_output
