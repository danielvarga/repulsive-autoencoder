import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input, BatchNormalization
from keras.layers.convolutional import Deconvolution2D
from keras.regularizers import l2

def generator_layers_wgan(latent_dim, batch_size, wd):
    layers = []
    layers.append(Reshape((1,1,latent_dim)))
    for channel, size, stride,border_mode in zip((4096, 2048, 1024, 512), (4, 8, 16, 32), (1, 2, 2, 2), ("valid", "same", "same", "same")):
        layers.append(Deconvolution2D(channel, 4, 4, output_shape=(batch_size, size, size, channel),
                                      subsample=(stride, stride), border_mode=border_mode, W_regularizer=l2(wd)))
        layers.append(BatchNormalization())
        layers.append(Activation('relu'))

    layers.append(Deconvolution2D(3, 4, 4, output_shape=(batch_size, 64, 64, 3),
                                  subsample=(2, 2), border_mode='same', W_regularizer=l2(wd)))
    layers.append(Activation('tanh'))
    return layers

class Decoder(object):
    pass

class DcganDecoder(Decoder):
    def __init__(self, latent_dim, batch_size, original_shape, wd):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.original_shape = original_shape
        assert original_shape == (64, 64, 3)
        self.wd = wd

    def __call__(self, recons_input):
        layers = generator_layers_wgan(self.latent_dim, self.batch_size, self.wd)

        generator_input = Input(batch_shape=(self.batch_size,self.latent_dim))
        generator_output = generator_input
        recons_output = recons_input
        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)

        return generator_input, recons_output, generator_output
