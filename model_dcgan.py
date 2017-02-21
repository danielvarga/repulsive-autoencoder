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
#channels = (128, 256, 512, 1024)
channels = (64, 128, 256, 512)

sizes = (64, 32, 16, 8, 4, 1)
strides = (2, 2, 2, 2, 2, 1)

encoder_activations = ("relu", "relu", "relu", "relu", "linear")
generator_activations = ("relu", "relu", "relu", "relu", "sigmoid")
use_bns = (True, True, True, True, False)

bn_epsilon = 0.00005

def normal_init(shape, name=None, dim_ordering="tf"):
    return initializations.normal(shape, scale=0.02, name=name, dim_ordering=dim_ordering)

def bn_beta_init(shape, name=None):
    return K.zeros(shape)

def bn_gamma_init(shape, name=None):
    return initializations.normal(shape, scale=0.02, name=name) + K.ones(shape)

def encoder_layers_wgan(latent_dim, batch_size, wd, bn_allowed, image_channel):
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
        if bn_allowed and use_bn: 
            layers.append(BatchNormalization(epsilon=bn_epsilon))
        layers.append(Activation(activation, name="encoder_{}".format(size)))
    layers.append(Reshape((latent_dim,)))
    return layers

def generator_layers_simple(latent_dim, batch_size, wd, bn_allowed, image_channel):
    layers = []
    layers.append(Dense(latent_dim, activation="relu"))
    layers.append(Dense(latent_dim, activation="relu"))
    layers.append(Dense(sizes[0] * sizes[0] * image_channel, activation="sigmoid"))
    layers.append(Reshape((sizes[0], sizes[0], image_channel)))
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
def generator_layers_wgan(latent_dim, batch_size, wd, bn_allowed, image_channel):
    generator_channels = list(reversed(channels)) + [image_channel]
    generator_sizes = reversed(sizes[:-1])
    generator_strides = reversed(strides[:-1])
    layers = []
    layers.append(Reshape((1,1,latent_dim)))
    import time
    for channel, size, stride, use_bn, activation in zip(generator_channels, generator_sizes, generator_strides, use_bns, generator_activations):
        if size == 4:
	    layers.append(Dense(4*4*latent_dim))
	    layers.append(Reshape((4,4,latent_dim)))
	    #layers.append(UpSampling2D(size=(4,4)))
	    #layers.append(UnPooling2D(poolsize=(4,4)))

            border_mode = "valid"
        else:
            layers.append(UpSampling2D(size=(2,2)))
	    #layers.append(UnPooling2D(poolsize=(2,2)))

            border_mode = "same"

	
#        layers.append(Deconvolution2D(channel, 4, 4, output_shape=(batch_size, size, size, channel), init=normal_init, bias=False,
#                                      subsample=(stride, stride), border_mode=border_mode, W_regularizer=l2(wd)))
	
        layers.append(Convolution2D(channel, 4, 4, init=normal_init, bias=False,
                                      subsample=(1, 1), border_mode="same", W_regularizer=l2(wd)))
	

        if bn_allowed and use_bn: 
            layers.append(BatchNormalization(epsilon=bn_epsilon))
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
        assert original_shape[:2] == (64, 64)
        self.wd = wd

    def __call__(self, x):
        layers = encoder_layers_wgan(self.latent_dim, self.batch_size, self.wd, self.original_shape[2])
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
        assert original_shape[:2] == (64, 64)
        self.wd = wd

    def __call__(self, recons_input):
        layers = generator_layers_wgan(self.latent_dim, self.batch_size, self.wd, self.original_shape[2])

        generator_input = Input(batch_shape=(self.batch_size,self.latent_dim))
        generator_output = generator_input
        recons_output = recons_input
        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)

        return generator_input, recons_output, generator_output

disc_channels = (64, 128, 256, 512, 1) 
#disc_channels = (8, 16, 32, 64, 1) 
disc_use_bns = (False, True, True, True, False)
disc_strides = (2, 2, 2, 2, 1)

def discriminator_layers_simple(latent_dim, wd, bn_allowed):
    layers = []
    layers.append(Flatten())
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
def discriminator_layers_wgan(latent_dim, wd, bn_allowed):
    alpha = 0.2
    layers=[]
    for channel, stride, use_bn in zip(disc_channels, disc_strides, disc_use_bns):
        if stride == 1:
            border_mode = "valid"
        else:
            border_mode = "same"
        layers.append(Convolution2D(channel, 4, 4, subsample=(stride, stride), border_mode=border_mode, init=normal_init, bias=False, W_regularizer=l2(wd)))
        if bn_allowed and use_bn:
            layers.append(BatchNormalization(epsilon=bn_epsilon))
        if stride != 1: 
            layers.append(LeakyReLU(alpha=alpha))
    layers.append(Reshape((1,)))
    return layers



### TODO: see if we need any of this code
# nc = 3            # # of channels in image
# npx = 64
# npy = 64          # # of pixels width/height of images
# nx = npx*npy*nc   # # of dimensions in X

# ngf = 512         # # of gen filters in first conv layer
# ndf = 128         # # of discrim filters in first conv layer
# # Ported from https://github.com/Newmu/dcgan_code/blob/master/faces/train_uncond_dcgan.py
# # 64x64
# def generator_layers():
#     layers = []
#     wd = WeightRegularizer(l2=args.wd)
#     assert npx % 16 == 0
#     assert npy % 16 == 0
#     layers.append(Dense(output_dim=ngf*8*(npx/16)*(npy/16), W_regularizer=wd))
#     layers.append(BatchNormalization())
#     layers.append(Activation('relu'))
#     layers.append(Reshape((npx/16, npy/16, ngf*8)))
#     deconv = False
#     for wide  in (4, 2, 1):
#         wd = WeightRegularizer(l2=args.wd)
#         if deconv:
#             layers.append(Deconvolution2D(ngf*wide, 5, 5,
#                                           output_shape=(args.batch_size, npx/wide/2, npy/wide/2, ngf*wide),
#                                           subsample=(2, 2), border_mode='same',
#                                           W_regularizer=wd))
#         else:
#             layers.append(UpSampling2D(size=(2, 2)))
#             layers.append(Convolution2D(ngf*wide, 5, 5, border_mode='same', W_regularizer=wd))
#             layers.append(BatchNormalization())
#             layers.append(Activation('relu'))

#     if deconv:
#         wd = WeightRegularizer(l2=args.wd)
#         layers.append(Deconvolution2D(nc, 5, 5, output_shape=(args.batch_size, nc, npx, npx),
#                                       subsample=(2, 2), border_mode='same', W_regularizer=wd))
#     else:
#         wd = WeightRegularizer(l2=args.wd)
#         layers.append(UpSampling2D(size=(2, 2)))
#         layers.append(Convolution2D(nc, 5, 5, border_mode='same', W_regularizer=wd))
#         layers.append(Activation('tanh'))
#     return layers


# # 64x64
# def discriminator_layers():
#     alpha = 0.2
#     layers=[]
#     wd = WeightRegularizer(l2=args.wd)
#     layers.append(Convolution2D(ndf, 5, 5, border_mode='same', W_regularizer=wd))
#     layers.append(LeakyReLU(alpha=alpha))
#     layers.append(BatchNormalization())
#     layers.append(Activation('relu'))
#     for wide in [2, 4, 8]:
#         wd = WeightRegularizer(l2=args.wd)
#         print "hey", ndf*wide
#         layers.append(Convolution2D(ndf*wide, 5, 5,
#                                     border_mode='same', subsample=(2, 2), W_regularizer=wd))
#         layers.append(BatchNormalization())
#         layers.append(LeakyReLU(alpha=alpha))
#     layers.append(Flatten())
#     wd = WeightRegularizer(l2=args.wd)
#     layers.append(Dense(1, activation='sigmoid', W_regularizer=wd))
#     return layers

# # From https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
# def generator_layer_mnist():
#     input = Input(shape=(100,))
#     net = input
#     net = Dense(input_dim=100, output_dim=1024)(net)
#     net = Activation('tanh')(net)
#     net = Dense(128*7*7)(net)
#     net = BatchNormalization()(net)
#     net = Activation('tanh')(net)
#     net = Reshape((128, 7, 7), input_shape=(128*7*7,))(net)
#     net = UpSampling2D(size=(2, 2))(net)
#     net = Convolution2D(64, 5, 5, border_mode='same')(net)
#     net = Activation('tanh')(net)
#     net = UpSampling2D(size=(2, 2))(net)
#     net = Convolution2D(1, 5, 5, border_mode='same')(net)
#     net = Activation('tanh')(net)
#     return input, net


# # https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
# def discriminator_layer_mnist():
#     input = Input(shape=(1, 28, 28))
#     net = input
#     net = Convolution2D(64, 5, 5,
#                         border_mode='same',
#                         input_shape=(1, 28, 28))(net)
#     net = Activation('tanh')(net)
#     net = MaxPooling2D(pool_size=(2, 2))(net)
#     net = Convolution2D(128, 5, 5)(net)
#     net = Activation('tanh')(net)
#     net = MaxPooling2D(pool_size=(2, 2))(net)
#     net = Flatten()(net)
#     net = Dense(1024)(net)
#     net = Activation('tanh')(net)
#     net = Dense(1)(net)
#     net = Activation('sigmoid')(net)
#     return input, net
