import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input, BatchNormalization, Flatten, UpSampling2D, Layer, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Deconvolution2D, Convolution2D

from keras.regularizers import l2
from keras import initializers

import keras.backend as K

from keras.engine.topology import Layer

kernel = 4 # do not change this or things might fail

def default_channels(model_type, model_size, last_channel):
    if model_size == "large":
        channels = [512, 1024, 2048, 4096]
        disc_channels = [64, 128, 256, 512, 1]
    elif model_size == "medium":
        channels = [256, 512, 1024, 2048]
        disc_channels = [32, 64, 128, 256, 1]
    elif model_size == "lsun":
        channels = [128, 256, 512, 1024]
        disc_channels = [32, 64, 128, 256, 1]
    elif model_size == "small":
        channels = [64, 128, 256, 512]
        disc_channels = [8, 16, 32, 64, 1] 
    elif model_size == "tiny":
        channels = [16, 32, 64, 128]
        disc_channels = [8, 16, 32, 64, 1] 
    else:
        assert False, "unknown model size " + model_size

    if model_type == "generator":
        return list(reversed(channels)) + [last_channel]
    elif model_type == "encoder":
        return channels + [last_channel]
    elif model_type =="discriminator":
        return disc_channels
    else:
        assert False, "unknown model_type"


def normal_init(shape, name=None, dim_ordering="tf"):
    return initializers.normal(shape, scale=0.02, name=name, dim_ordering=dim_ordering)

def bn_beta_init(shape, name=None):
    return K.zeros(shape)

def bn_gamma_init(shape, name=None):
    return initializers.normal(shape, scale=0.02, name=name) + K.ones(shape)

def encoder_layers_wgan(channels, wd, bn_allowed):
    layers=[]
    for idx, channel in enumerate(channels):
        if idx == (len(channels)-1): # stride is 2 except for the last layer where it is 1
            border_mode="valid"
            stride=1
            activation = "linear"
            use_bn = False
        else:
            border_mode="same"
            stride=2
            activation="relu"
            use_bn = bn_allowed
        layers.append(Convolution2D(channel, (kernel, kernel), strides=(stride, stride), padding=border_mode, use_bias=False, kernel_regularizer=l2(wd)))
        if use_bn: 
            layers.append(BatchNormalization(axis=-1))
        layers.append(Activation(activation, name="encoder_{}".format(idx)))
    layers.append(Flatten())
    return layers

def generator_layers_dense(latent_dim, batch_size, wd, bn_allowed, image_shape):
    layers = []
    layers.append(Dense(latent_dim, activation="relu"))
    layers.append(Dense(latent_dim, activation="relu"))
    layers.append(Dense(np.prod(image_shape), activation="sigmoid"))
    layers.append(Reshape(image_shape))
    return layers

"""    
DCGAN_G (
  (main): Sequential (
    (initial.100-4096.convt): ConvTranspose2d(100, 4096, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (initial.4096.batchnorm): BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True)
    (initial.4096.relu): ReLU (inplace)
    (pyramid.4096-2048.convt): ConvTranspose2d(4096, 2048, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid.2048.batchnorm): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True)
    (pyramid.2048.relu): ReLU (inplace)
    (pyramid.2048-1024.convt): ConvTranspose2d(2048, 1024, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid.1024.batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True)
    (pyramid.1024.relu): ReLU (inplace)
    (pyramid.1024-512.convt): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid.512.batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (pyramid.512.relu): ReLU (inplace)
    (final.512-3.convt): ConvTranspose2d(512, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (final.3.tanh): Tanh ()
  )
)
"""
def generator_layers_wgan(channels, latent_dim, wd, bn_allowed, batch_size, firstX=1, firstY=1):
    layers = []
    assert latent_dim % (firstX * firstY) == 0
    layers.append(Reshape((firstX,firstY,latent_dim/firstX/firstY)))
    sizeX = firstX
    sizeY = firstY
    stride=2
    for idx, channel in enumerate(channels):
        if idx == 0:
            sizeX *= 4
            sizeY *= 4
            border_mode="valid"
        else:
            sizeX *= 2
            sizeY *= 2
            border_mode="same"
        if idx == (len(channels)-1):
            activation="sigmoid"
            use_bn = False
        else:
            activation="relu"
            use_bn = bn_allowed
        layers.append(Deconvolution2D(channel, (kernel, kernel), use_bias=False, #output_shape=(batch_size, sizeX, sizeY, channel)
                                      strides=(stride, stride), padding=border_mode, kernel_regularizer=l2(wd)))
        if use_bn: 
            layers.append(BatchNormalization(axis=-1))
        layers.append(Activation(activation, name="generator_{}".format(idx)))
    return layers

class Encoder(object):
    pass

class DcganEncoder(Encoder):
    def __init__(self, args):
        self.latent_dim = args.latent_dim
        self.wd = args.encoder_wd
        self.bn_allowed = args.encoder_use_bn
        self.original_shape = args.original_shape
        self.dcgan_size = args.dcgan_size
        self.channels = default_channels("encoder", self.dcgan_size, self.latent_dim)
        reduction = 2 ** (len(self.channels)+1)
        assert self.original_shape[0] % reduction == 0
        assert self.original_shape[1] % reduction == 0

    def __call__(self, x):
        layers = encoder_layers_wgan(self.channels, self.wd, self.bn_allowed)
        h = x
        for layer in layers:
            h = layer(h)
        return h

class Decoder(object):
    pass

class DcganDecoder(Decoder):
    def __init__(self, args):
        self.latent_dim = args.latent_dim
        self.wd = args.decoder_wd
        self.bn_allowed = args.decoder_use_bn
        self.batch_size = args.batch_size
        self.original_shape = args.original_shape
        self.dcgan_size = args.dcgan_size
        self.channels = default_channels("generator", self.dcgan_size, self.original_shape[2])
        reduction = 2 ** (len(self.channels)+1)
        assert self.original_shape[0] % reduction == 0
        assert self.original_shape[1] % reduction == 0
        self.firstX = self.original_shape[0] // reduction
        self.firstY = self.original_shape[1] // reduction


    def __call__(self, recons_input):
        layers = generator_layers_wgan(self.channels, self.latent_dim, self.wd, self.bn_allowed, self.batch_size, self.firstX, self.firstY)

        generator_input = Input(batch_shape=(self.batch_size,self.latent_dim))
        generator_output = generator_input
        recons_output = recons_input
        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)

        return generator_input, recons_output, generator_output


def discriminator_layers_dense(wd, bn_allowed):
    layers = []
    layers.append(Flatten())
    layers.append(Dense(100, activation="relu"))
    layers.append(Dense(100, activation="relu"))
    layers.append(Dense(100, activation="relu"))
    layers.append(Dense(100, activation="relu"))
    layers.append(Dense(100, activation="relu"))
    layers.append(Dense(1))
    return layers

"""
DCGAN_D (
  (main): Sequential (
    (initial.conv.3-64): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (initial.relu.64): LeakyReLU (0.2, inplace)
    (pyramid.64-128.conv): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid.128.batchnorm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (pyramid.128.relu): LeakyReLU (0.2, inplace)
    (pyramid.128-256.conv): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid.256.batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (pyramid.256.relu): LeakyReLU (0.2, inplace)
    (pyramid.256-512.conv): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid.512.batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (pyramid.512.relu): LeakyReLU (0.2, inplace)
    (final.512-1.conv): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
"""
def discriminator_layers_wgan(channels, wd, bn_allowed):
    alpha = 0.2
    layers=[]
    for idx, channel in enumerate(channels):
        if idx == (len(channels)-1): # stride is 2 except for the last layer where it is 1
            border_mode="valid"
            stride=1
            use_bn = False
            nonlinearity = False
        else:
            border_mode="same"
            stride=2
            if idx == 0:
                use_bn = False
            else:
                use_bn = bn_allowed
            nonlinearity = True
        layers.append(Convolution2D(channel, (kernel, kernel), strides=(stride, stride), padding=border_mode, use_bias=False, kernel_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization(axis=-1))
        if nonlinearity: 
            layers.append(LeakyReLU(alpha=alpha))
    layers.append(Reshape((1,)))
    return layers

