'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

from keras.layers import Input, Dense, Lambda, Convolution2D, Deconvolution2D, Reshape, Flatten, ZeroPadding2D, merge, Activation, Layer, AveragePooling2D, Merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import *
from keras.regularizers import l1, l2

import tensorflow as tf


add_tables=[]
decoder_layers = []


def discnet_encoder_drop(nb_filter=32, act="relu", weights_init=None, biases_init=None, use_bias=False, wd=0.003, batch_size=32):

    if weights_init is None:
	#weights_init = normal(shape, scale)
	pass

    if biases_init is None:
	#biases_init = 0.0
	pass


    layers = []

    bn_axis = 3

    conv_1 = Convolution2D(nb_filter, 3, 3, subsample=(1,1), border_mode="same", W_regularizer=l2(wd), bias=use_bias)
    layers.append(conv_1)

    bn_1 = BatchNormalization(axis=bn_axis)
    layers.append(bn_1)

    act_1 = Activation(act)
    layers.append(act_1)

    conv_2 = Convolution2D(nb_filter, 3, 3, subsample=(1,1), border_mode="same", W_regularizer=l2(wd), bias=use_bias)
    layers.append(conv_2)

    bn_2 = BatchNormalization(axis=bn_axis)
    layers.append(bn_2)

    act_2 = Activation(act)
    layers.append(act_2)

    deconv_3 = Convolution2D(nb_filter, 2, 2,
                                           #output_shape,
                                           border_mode='same',
                                           subsample=(2,2),
                                           activation='linear',
                                           W_regularizer=l2(wd), bias=use_bias) # todo
    layers.append(deconv_3)

    bn_3 = BatchNormalization(axis=bn_axis)
    layers.append(bn_3)

    act_3 = Activation(act)
    layers.append(act_3)

    return layers




def discnet_decoder_drop(image_size, nb_filter=32, act="relu", weights_init=None, biases_init=None, use_bias=False, wd=0.003, batch_size=32):

    if weights_init is None:
	#weights_init = normal(shape, scale)
	pass

    if biases_init is None:
	#biases_init = 0.0
	pass


    layers = []

    bn_axis = 3

    conv_1 = Convolution2D(nb_filter, 3, 3, subsample=(1,1), border_mode="same", W_regularizer=l2(wd), bias=use_bias)
    layers.append(conv_1)

    bn_1 = BatchNormalization(axis=bn_axis, mode=2)
    layers.append(bn_1)

    act_1 = Activation(act)
    layers.append(act_1)

    conv_2 = Convolution2D(nb_filter, 3, 3, subsample=(1,1), border_mode="same", W_regularizer=l2(wd), bias=use_bias)
    layers.append(conv_2)

    bn_2 = BatchNormalization(axis=bn_axis, mode=2)
    layers.append(bn_2)

    act_2 = Activation(act)
    layers.append(act_2)

    deconv_3 = Deconvolution2D(nb_filter, 2, 2,
                                           output_shape=(batch_size, image_size[0]//2, image_size[1]//2, nb_filter),
                                           border_mode='same',
                                           subsample=(2,2),
                                           activation='linear',
                                           W_regularizer=l2(wd), bias=use_bias) # todo
    layers.append(deconv_3)

    bn_3 = BatchNormalization(axis=bn_axis, mode=2)
    layers.append(bn_3)

    act_3 = Activation(act)
    layers.append(act_3)

    return layers


class Encoder(object):
    pass

class ConvEncoder(Encoder):
    def __init__(self, depth, latent_dim, encoder_filters, image_dims, batch_size=32, wd=0.003):
        self.depth = depth
        self.latent_dim = latent_dim
	self.encoder_filters = encoder_filters
	self.image_dims = image_dims
        self.batch_size = batch_size
        self.wd = wd

    def __call__(self, x):

	layers = []

	reshape = Reshape(self.image_dims)
	layers.append(reshape)

	for d in range(self.depth):
	    layers += discnet_encoder_drop(nb_filter=32*(2**d), batch_size=self.batch_size)

	layers.append(Flatten())

	# Decoder MLP
	intermediate_dims = [self.encoder_filters, 1000, self.latent_dim]
        for dim in intermediate_dims:
	    dense = Dense(dim, activation="relu")
	    layers.append(dense)

        for layer in layers:
            x = layer(x)

        z = x
        return z


class Decoder(object):
    pass # TODO interface

class ConvDecoder(Decoder):

    def __init__(self, depth, latent_dim, encoder_filters, image_dims, batch_size=32, wd=0.003):
        self.depth = depth
        self.latent_dim = latent_dim
	self.encoder_filters = encoder_filters
	self.image_dims = image_dims
        self.batch_size = batch_size
        self.wd = wd

    def __call__(self, z):

	layers = []

	# Decoder MLP
	self.encoder_filters = self.image_dims[0]//(2**(self.depth-1)) * self.image_dims[1]//(2**(self.depth-1)) * 1
        
	#intermediate_dims = [self.encoder_filters, 1000, self.latent_dim]
	intermediate_dims = [self.latent_dim, 1000, self.encoder_filters]

        for dim in intermediate_dims:
	    dense = Dense(dim, activation="relu")
	    layers.append(dense)

        image_size = (self.image_dims[0]//(2**(self.depth-1)), self.image_dims[1]//(2**(self.depth-1)), 1)
        print(image_size)
	layers.append(Reshape(image_size))

	for d in range(self.depth, -1, -1):
	    image_size = (self.image_dims[0]//(2**d), self.image_dims[1]//(2**d))
	    print(image_size)
	    layers += discnet_decoder_drop(image_size=image_size, nb_filter=32*(2**d), batch_size=self.batch_size)

	# Make the picture
	conv_out = Convolution2D(1, 1, 1, subsample=(1,1), border_mode="same")
        layers.append(conv_out)

	logistic = Activation("sigmoid")
	layers.append(logistic)

        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        decoder_input = Input(batch_shape=(self.batch_size, self.latent_dim,))

	x = decoder_input
        for layer in layers:
	    print(layer)
            x = layer(x)
        _x_decoded_mean = x

	x = z
        for layer in layers:
            x = layer(x)
        x_decoded_mean = x

        return decoder_input, x_decoded_mean, _x_decoded_mean
