'''
Convolutional encoder/decoder that is symmetrical
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

if K.image_dim_ordering() == 'tf':
    feature_axis = 3
elif K.image_dim_ordering() == 'th':
    feature_axis = 1

def discnet_encoder_drop(depth, base_filter_nums, batch_size, use_bias=False, wd=0.003):

    layers = []
    layer_count = len(base_filter_nums)
    for i in range(layer_count):
        nb_filter = base_filter_nums[i] * (2**depth)
        if i < (layer_count - 1):
            nb_row = 3
            nb_col = 3
            subsample = (1,1)
        else:
            nb_row = 2
            nb_col = 2
            subsample = (2,2)
        if i == 0:
            act = "sigmoid"
        else:
            act = "relu"

        conv = Convolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=subsample, W_regularizer=l2(wd), bias=use_bias,
                             name="enc_conv_{}_{}".format(depth,i))
        layers.append(conv)

        # bn = BatchNormalization(axis=feature_axis)
        # layers.append(bn)

        activ = Activation(act, name="enc_act_{}_{}".format(depth,i))
        layers.append(activ)
    return layers


def discnet_decoder_drop(image_size, depth, base_filter_nums, batch_size, use_bias=False, wd=0.003):

    layers = []
    layer_count = len(base_filter_nums)
    image_filters = image_size[2]
    filter_nums = [image_filters] + list(base_filter_nums[:layer_count-1])
    for i in reversed(range(layer_count)):
        nb_filter = filter_nums[i] * (2**depth)
        if i < (layer_count - 1):
            deconv = Convolution2D(nb_filter, 3, 3, subsample=(1,1), border_mode="same", W_regularizer=l2(wd), bias=use_bias,
                                   name="dec_conv_{}_{}".format(depth,i))
        else:
            if K.image_dim_ordering() == 'tf':
                output_shape = (batch_size, image_size[0], image_size[1], nb_filter)
            elif K.image_dim_ordering() == 'th':
                output_shape = (batch_size, nb_filter, image_size[1], image_size[2])
            deconv = Deconvolution2D(nb_filter, 2, 2, output_shape=output_shape, border_mode='same', subsample=(2,2), 
                                     W_regularizer=l2(wd), bias=use_bias,
                                     name="dec_conv_{}_{}".format(depth,i))
        layers.append(deconv)

        # bn = BatchNormalization(axis=feature_axis, mode=2)
        # layers.append(bn)

        if i == 0:
            act = "sigmoid"
        else:
            act = "relu"
        activ = Activation(act, name="dec_act_{}_{}".format(depth,i))
        layers.append(activ)
    return layers


class Encoder(object):
    pass

class ConvEncoder(Encoder):
    def __init__(self, depth, latent_dim, intermediate_dims, image_dims, base_filter_nums, batch_size, wd=0.003):
        self.depth = depth
        self.latent_dim = latent_dim
	self.intermediate_dims = intermediate_dims
	self.image_dims = image_dims
        self.batch_size = batch_size
	self.base_filter_nums = base_filter_nums
        self.wd = wd

    def __call__(self, x):


	layers = []

#	reshape = Reshape(self.image_dims)
#	layers.append(reshape)

	for d in range(self.depth):
	    layers += discnet_encoder_drop(d, self.base_filter_nums, self.batch_size)

	layers.append(Flatten())

	# Decoder MLP
	## bn rect, 1st, linear 2nd act
        for dim in self.intermediate_dims:
	    dense = Dense(dim)
	    layers.append(dense)

#	    bn = BatchNormalization(mode=2)
#	    layers.append(bn)

	    act = Activation("relu")
	    layers.append(act)


        for layer in layers:
            x = layer(x)
        return x


class Decoder(object):
    pass # TODO interface

class ConvDecoder(Decoder):

    def __init__(self, depth, latent_dim, intermediate_dims, image_dims, base_filter_nums, batch_size, wd=0.003):
        self.depth = depth
        self.latent_dim = latent_dim
	self.intermediate_dims = intermediate_dims
	self.image_dims = image_dims
        self.batch_size = batch_size
	self.base_filter_nums = base_filter_nums
        self.wd = wd

    def __call__(self, z):

	layers = []

	# Decoder MLP
        image_size = (self.image_dims[0]//(2**(self.depth)), self.image_dims[1]//(2**(self.depth)), self.base_filter_nums[0] * (2**self.depth))
        self.encoder_filters = np.prod(image_size)

	intermediate_dims = self.intermediate_dims + [self.encoder_filters]
        for dim in intermediate_dims:
	    dense = Dense(dim, activation="relu")
	    layers.append(dense)
	layers.append(Reshape(image_size))

	for d in range(self.depth-1, -1, -1):
            if d == 0:
                nb_features = self.image_dims[2]
            else:
                nb_features = self.base_filter_nums[0] * (2**(d-1))

	    image_size = (self.image_dims[0]//(2**d), self.image_dims[1]//(2**d), nb_features)
	    # print(image_size)
	    layers += discnet_decoder_drop(image_size, d, self.base_filter_nums, self.batch_size)

	# Make the picture
#	conv_out = Convolution2D(1, 1, 1, subsample=(1,1), border_mode="same")
#        layers.append(conv_out)
#	logistic = Activation("sigmoid")
#	layers.append(logistic)

#	layers.append(Flatten())

        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        decoder_input = Input(batch_shape=(self.batch_size, self.latent_dim,))

	x = z
        _x = decoder_input
        for layer in layers:
            x = layer(x)
            _x = layer(_x)
        x_decoded_mean = x
        _x_decoded_mean = _x

        return decoder_input, x_decoded_mean, _x_decoded_mean
