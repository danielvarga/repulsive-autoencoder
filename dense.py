import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input
from keras.regularizers import l2

class Encoder(object):
    pass

class Decoder(object):
    pass

class DenseEncoder(Encoder):
    def __init__(self, intermediate_dims, activation, wd):
        self.intermediate_dims = intermediate_dims
        self.activation = activation
        self.wd = wd

    def __call__(self, x):
        h = Flatten()(x)
        for intermediate_dim in self.intermediate_dims:
            h = Dense(intermediate_dim, W_regularizer=l2(self.wd))(h)
            h = Activation(self.activation)(h)
        return h

class DenseDecoder(Decoder):
    def __init__(self, latent_dim, intermediate_dims, original_shape, activation, wd):
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.original_shape = original_shape
        self.activation = activation
        self.wd = wd

    def __call__(self, recons_input):
        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        layers = []
        for intermediate_dim in reversed(self.intermediate_dims):
            layers.append(Dense(intermediate_dim, W_regularizer=l2(self.wd)))
            layers.append(Activation(self.activation))
        layers.append(Dense(np.prod(self.original_shape), activation='sigmoid', name="decoder_top", W_regularizer=l2(self.wd)))
        layers.append(Reshape(self.original_shape))

        generator_input = Input(shape=(self.latent_dim,))
        generator_output = generator_input
        recons_output = recons_input
        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)

        return generator_input, recons_output, generator_output
