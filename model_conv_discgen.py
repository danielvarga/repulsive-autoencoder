'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

from keras.layers import Input, Dense, Lambda, Convolution2D, Deconvolution2D, Reshape, Flatten, ZeroPadding2D, merge, Activation, Layer, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import *
from keras.regularizers import l1, l2
import activations

add_tables=[]
decoder_layers = []


def discnet_encoder_drop(nb_filter=32, act=activations.activation, weights_init=None, biases_init=None, use_bias=False, wd=0.003, batch_size=32):

    if weights_init is None:
        #weights_init = normal(shape, scale)
        pass

    if biases_init is None:
        #biases_init = 0.0
        pass


    layers = []

    bn_axis = 3

    conv_1 = Convolution2D(nb_filter, (3, 3), strides=(1,1), padding="same", kernel_regularizer=l2(wd), use_bias=use_bias)
    layers.append(conv_1)

    bn_1 = BatchNormalization(axis=bn_axis)
    layers.append(bn_1)

    act_1 = Activation(act)
    layers.append(act_1)

    conv_2 = Convolution2D(nb_filter, (3, 3), strides=(1,1), padding="same", kernel_regularizer=l2(wd), use_bias=use_bias)
    layers.append(conv_2)

    bn_2 = BatchNormalization(axis=bn_axis)
    layers.append(bn_2)

    act_2 = Activation(act)
    layers.append(act_2)

    deconv_3 = Convolution2D(nb_filter, (2, 2),
                                           #output_shape,
                                           padding='same',
                                           strides=(2,2),
                                           activation='linear',
                                           kernel_regularizer=l2(wd), use_bias=use_bias) # todo
    layers.append(deconv_3)

    bn_3 = BatchNormalization(axis=bn_axis)
    layers.append(bn_3)

    act_3 = Activation(act)
    layers.append(act_3)

    return layers




def discnet_decoder_drop(image_size, nb_filter=32, act=activations.activation, weights_init=None, biases_init=None, use_bias=False, wd=0.0, batch_size=32, use_bn=False):

    if weights_init is None:
        #weights_init = normal(shape, scale)
        pass

    if biases_init is None:
        #biases_init = 0.0
        pass


    layers = []

    bn_axis = 3

    conv_1 = Convolution2D(nb_filter, (3, 3), strides=(1,1), padding="same", kernel_regularizer=l2(wd), use_bias=use_bias)
    layers.append(conv_1)

    if use_bn:
        bn_1 = BatchNormalization(axis=bn_axis)
        layers.append(bn_1)

    act_1 = Activation(act)
    layers.append(act_1)

    conv_2 = Convolution2D(nb_filter, (3, 3), strides=(1,1), padding="same", kernel_regularizer=l2(wd), use_bias=use_bias)
    layers.append(conv_2)

    if use_bn:
        bn_2 = BatchNormalization(axis=bn_axis)
        layers.append(bn_2)

    act_2 = Activation(act)
    layers.append(act_2)

    deconv_3 = Deconvolution2D(nb_filter, (2, 2),
                                           #output_shape=(batch_size, image_size[0], image_size[1], nb_filter),
                                           padding='same',
                                           strides=(2,2),
                                           activation='linear',
                                           kernel_regularizer=l2(wd), use_bias=use_bias) # todo
    layers.append(deconv_3)

    if use_bn:
        bn_3 = BatchNormalization(axis=bn_axis)
        layers.append(bn_3)

    act_3 = Activation(act)
    layers.append(act_3)

    return layers


class Encoder(object):
    pass

class ConvEncoder(Encoder):
    def __init__(self, depth, latent_dim, intermediate_dims, image_dims, batch_size=32, base_filter_num=32, wd=0.003):
        self.depth = depth
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.image_dims = image_dims
        self.batch_size = batch_size
        self.base_filter_num = base_filter_num
        self.wd = wd

    def __call__(self, x):


        layers = []

#        reshape = Reshape(self.image_dims)
#        layers.append(reshape)

        for d in range(self.depth):
            layers += discnet_encoder_drop(nb_filter=self.base_filter_num*(2**d), batch_size=self.batch_size)

        layers.append(Flatten())

        # Decoder MLP
        bn_axis = 3
        ## bn rect, 1st, linear 2nd act
        for dim in self.intermediate_dims:
            dense = Dense(dim)
            layers.append(dense)

            bn = BatchNormalization()
            layers.append(bn)

            act = Activation(activations.activation)
            layers.append(act)


        for layer in layers:
            x = layer(x)

        z = x
        return z


class Decoder(object):
    pass # TODO interface

class ConvDecoder(Decoder):

    def __init__(self, depth=3, latent_dim=512, intermediate_dims=None, image_dims=None, batch_size=32, base_filter_num=32, wd=0.0, use_bn=False):
        self.depth = depth
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.image_dims = image_dims
        self.batch_size = batch_size
        self.base_filter_num = base_filter_num
        self.wd = wd
        self.use_bn = use_bn

    def __call__(self, z):

        layers = []

        # Decoder MLP
        self.encoder_filters = self.image_dims[0]//(2**(self.depth)) * self.image_dims[1]//(2**(self.depth)) * self.base_filter_num * (2**(self.depth-1))

        intermediate_dims = self.intermediate_dims + [self.encoder_filters]
        for dim in intermediate_dims:
            dense = Dense(dim)
            layers.append(dense)
            layers.append(Activation(activations.activation))

        image_size = (self.image_dims[0]//(2**(self.depth)), self.image_dims[1]//(2**(self.depth)), self.base_filter_num * (2**(self.depth-1)))
        layers.append(Reshape(image_size))


        for d in range(self.depth-1, -1, -1):
            image_size = (self.image_dims[0]//(2**d), self.image_dims[1]//(2**d))
            print(image_size)
            layers += discnet_decoder_drop(image_size=image_size, nb_filter=self.base_filter_num*(2**d), batch_size=self.batch_size, wd=self.wd, use_bn=self.use_bn)

        # Make the picture
        conv_out = Convolution2D(self.image_dims[2], (1, 1), strides=(1,1), padding="same")
        layers.append(conv_out)

        logistic = Activation("sigmoid")
        layers.append(logistic)

#        layers.append(Flatten())

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
