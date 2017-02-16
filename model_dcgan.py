import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input, BatchNormalization, Flatten, UpSampling2D, Layer, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Deconvolution2D, Convolution2D

from keras.regularizers import l2
from keras import initializations

import keras.backend as K

from keras.engine.topology import Layer

# image_dim is missing from the beginning, latent_dim is missing from the end
#channels = (512, 1024, 2048, 4096)
channels = (128, 256, 512, 1024)
#channels = (64, 128, 256, 512)

sizes = (64, 32, 16, 8, 4, 1)
strides = (2, 2, 2, 2, 2, 1)

encoder_activations = ("relu", "relu", "relu", "relu", "linear")
generator_activations = ("relu", "relu", "relu", "relu", "tanh")
use_bns = (True, True, True, True, False)

bn_epsilon = 0.00005

def normal_init(shape, name=None, dim_ordering="tf"):
    return initializations.normal(shape, scale=0.02, name=name, dim_ordering=dim_ordering)

def bn_beta_init(shape, name=None):
    return K.zeros(shape)

def bn_gamma_init(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name) + K.ones(shape)

def encoder_layers_wgan(latent_dim, batch_size, wd, image_channel):
    encoder_channels = list(channels) + [latent_dim]
    encoder_sizes = sizes[1:]
    encoder_strides = strides[1:]
    layers=[]
    for channel, size, stride, use_bn, activation in zip(encoder_channels, encoder_sizes, encoder_strides, use_bns, encoder_activations):
        if stride == 1:
            border_mode = "valid"
        else:
            border_mode = "same"
        layers.append(Convolution2D(channel, 4, 4, subsample=(stride, stride), border_mode=border_mode, init=normal_init, bias=False, W_regularizer=l2(wd)))
        if use_bn: layers.append(BatchNormalization(epsilon=bn_epsilon))
        layers.append(Activation(activation, name="encoder_{}".format(size)))
    layers.append(Reshape((latent_dim,)))
    return layers

def generator_layers_wgan(latent_dim, batch_size, wd, image_channel):
    generator_channels = list(reversed(channels)) + [image_channel]
    generator_sizes = reversed(sizes[:-1])
    generator_strides = reversed(strides[:-1])
    layers = []
    layers.append(Reshape((1,1,latent_dim)))
    import time
    for channel, size, stride, use_bn, activation in zip(generator_channels, generator_sizes, generator_strides, use_bns, generator_activations):
        if size == 4:
	    #layers.append(Dense(4*4*latent_dim))
	    #layers.append(Reshape((4,4,latent_dim)))
	    layers.append(UpSampling2D(size=(4,4)))
	    #layers.append(UnPooling2D(poolsize=(4,4)))

            border_mode = "valid"
        else:
            layers.append(UpSampling2D(size=(2,2)))
	    #layers.append(UnPooling2D(poolsize=(2,2)))

            border_mode = "same"

	"""
        layers.append(Deconvolution2D(channel, 4, 4, output_shape=(batch_size, size, size, channel), init=normal_init, bias=False,
                                      subsample=(stride, stride), border_mode=border_mode, W_regularizer=l2(wd)))
	"""
        layers.append(Convolution2D(channel, 4, 4, init=normal_init, bias=False,
                                      subsample=(1, 1), border_mode="same", W_regularizer=l2(wd)))
	

        if use_bn: layers.append(BatchNormalization(epsilon=bn_epsilon))
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

#disc_channels = (64, 128, 256, 512, 1) 
disc_channels = (8, 16, 32, 64, 1) 
disc_use_bns = (False, True, True, True, False)
disc_strides = (2, 2, 2, 2, 1)

def discriminator_layers_wgan(latent_dim, wd):
    alpha = 0.2
    layers=[]
    for channel, stride, use_bn in zip(disc_channels, disc_strides, disc_use_bns):
        if stride == 1:
            border_mode = "valid"
        else:
            border_mode = "same"
        layers.append(Convolution2D(channel, 4, 4, subsample=(stride, stride), border_mode=border_mode, init=normal_init, bias=False, W_regularizer=l2(wd)))
        if use_bn: layers.append(BatchNormalization(epsilon=bn_epsilon))
        if stride != 1: layers.append(LeakyReLU(alpha=alpha))
    layers.append(Reshape((1,)))
    return layers
