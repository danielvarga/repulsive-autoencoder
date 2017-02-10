import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input, BatchNormalization, Flatten
from keras.layers.convolutional import Deconvolution2D, Convolution2D
from keras.regularizers import l2

channels = (3, 512, 1024, 2048, 4096) # latent_dim is missing from the end of this tuple
sizes = (64, 32, 16, 8, 4, 1)
strides = (2, 2, 2, 2, 2, 1)

encoder_activations = ("relu", "relu", "relu", "relu", "linear")
generator_activations = ("relu", "relu", "relu", "relu", "tanh")
use_bns = (True, True, True, True, False)

def encoder_layers_wgan(latent_dim, batch_size, wd):
    encoder_channels = list(channels[1:]) + [latent_dim]
    encoder_sizes = sizes[1:]
    encoder_strides = strides[1:]
    layers=[]
    for channel, size, stride, use_bn, activation in zip(encoder_channels, encoder_sizes, encoder_strides, use_bns, encoder_activations):
        if stride == 1:
            border_mode = "valid"
        else:
            border_mode = "same"
        layers.append(Convolution2D(channel, 4, 4, subsample=(stride, stride), border_mode=border_mode, W_regularizer=l2(wd)))
        if use_bn: layers.append(BatchNormalization())
        layers.append(Activation(activation, name="encoder_{}".format(size)))
    layers.append(Reshape((latent_dim,)))
    return layers

def generator_layers_wgan(latent_dim, batch_size, wd):
    generator_channels = reversed(channels)
    generator_sizes = reversed(sizes[:-1])
    generator_strides = reversed(strides[:-1])
    layers = []
    layers.append(Reshape((1,1,latent_dim)))
    for channel, size, stride, use_bn, activation in zip(generator_channels, generator_sizes, generator_strides, use_bns, generator_activations):
        if stride == 1:
            border_mode = "valid"
        else:
            border_mode = "same"
        layers.append(Deconvolution2D(channel, 4, 4, output_shape=(batch_size, size, size, channel),
                                      subsample=(stride, stride), border_mode=border_mode, W_regularizer=l2(wd)))
        if use_bn: layers.append(BatchNormalization())
        layers.append(Activation(activation, name="generator_{}".format(size)))
    return layers

class Encoder(object):
    pass

class DcganEncoder(Encoder):
    def __init__(self, latent_dim, batch_size, original_shape, wd):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.original_shape = original_shape
        print original_shape
        assert original_shape == (64, 64, 3)
        self.wd = wd

    def __call__(self, x):
        layers = encoder_layers_wgan(self.latent_dim, self.batch_size, self.wd)
        h = x
        for layer in layers:
            h = layer(h)
        return h

class Decoder(object):
    pass

class DcganDecoder(Decoder):
    def __init__(self, latent_dim, batch_size, original_shape, wd):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.original_shape = original_shape
        print original_shape
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
