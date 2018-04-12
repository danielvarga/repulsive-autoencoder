'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

from keras.layers import Input, Dense, Lambda, Convolution2D, Deconvolution2D, Reshape, Flatten, ZeroPadding2D, merge
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import *
from keras.regularizers import l1, l2

class Encoder(object):
    pass

class ConvEncoder(Encoder):
    def __init__(self, intermediate_dim, img_size):
        self.intermediate_dim = intermediate_dim
        self.img_size = img_size

    def __call__(self, x):
        wd = 0.003
        nb_conv=3
        nb_filters = 64
        batch_size = 1000
        image_chns = 1
        xr = Reshape(self.img_size)(x)
        conv_1 = Convolution2D(nb_filters, (nb_conv, nb_conv), padding='same', activation='relu', kernel_regularizer=l2(wd))(xr)
        conv_2 = Convolution2D(nb_filters, (nb_conv, nb_conv),
                               padding='same', activation='relu',
                               strides=(2, 2), kernel_regularizer=l2(wd))(conv_1)
        conv_3 = Convolution2D(nb_filters, (nb_conv, nb_conv),
                               padding='same', activation='relu',
                               strides=(1, 1), kernel_regularizer=l2(wd))(conv_2)

        conv_4 = Convolution2D(nb_filters, (nb_conv, nb_conv),
                               padding='same', activation='relu',
                               strides=(1, 1), kernel_regularizer=l2(wd))(conv_3)
        flat = Flatten()(conv_4)
        d1 = Dense(self.intermediate_dim)(flat)
        return d1


class Decoder(object):
    pass # TODO interface

class ConvDecoder(Decoder):
    def __init__(self, latent_dim, intermediate_dim, original_dim, img_size):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.original_dim = original_dim
        self.img_size = img_size

    def __call__(self, z):

        wd = 0.003
        nb_conv=3
        nb_filters = 64
        batch_size = 1000

        img_chns = 1
        h = self.img_size[0]
        w = self.img_size[1]
        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        decoder_input = Input(shape=(self.latent_dim,))

        h_decoded = z
        _h_decoded = decoder_input

        decoder_hid = Dense(self.intermediate_dim)
        decoder_upsample = Dense(self.intermediate_dim * 4, W_regularizer=l2(wd))
        decoder_upsample2 = Dense(4 * (h//2) * (w//2), W_regularizer=l2(wd))
        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, 4, h//2, w//2)
        else:
            output_shape = (batch_size, h//2, w//2, 4)

        decoder_reshape = Reshape(output_shape[1:])

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, h//2, w//2)
        else:
            output_shape = (batch_size, h//2, w//2, nb_filters)

        decoder_deconv_1 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=(1, 1),
                                           activation='relu',
                                           kernel_regularizer=l2(wd))

        decoder_deconv_2 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=(1, 1),
                                           activation='relu')

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, h-1, w-1)
        else:
            output_shape = (batch_size, h-1, w-1, nb_filters)

        decoder_deconv_3_upsamp = Deconvolution2D(nb_filters, (3, 3),
                                                  padding='same',
                                                  strides=(2, 2),
                                                  activation='relu', name="upsz", kernel_regularizer=l2(wd))

        decoder_deconv_4 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=(1, 1),
                                           activation='relu')

        decoder_mean_squash = Convolution2D(img_chns, (2, 2),
                                            padding='same',
                                            activation='sigmoid', name='reconv_flatten', kernel_regularizer=l2(wd))

        zp = ZeroPadding2D({'top_pad':1, 'left_pad':1, 'right_pad':0, 'bottom_pad':0})


        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        up_decoded = decoder_upsample2(up_decoded)
        #up_decoded = decoder_upsample3(up_decoded)

        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        deconv_3_decoded = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_relu = merge([deconv_3_decoded, decoder_deconv_4(deconv_3_decoded)], mode="sum")

        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        x_decoded_mean_np = x_decoded_mean_squash
        x_decoded_mean = zp(x_decoded_mean_np)
        x_decoded_mean = Flatten()(x_decoded_mean)



        # build a digit generator that can sample from the learned distribution

        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _up_decoded = decoder_upsample2(_up_decoded)
        #_up_decoded = decoder_upsample3(_up_decoded)

        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _deconv_3_decoded = decoder_deconv_3_upsamp(_deconv_2_decoded)
        #_x_decoded_relu = _deconv_3_decoded + decoder_deconv_4(_deconv_3_decoded)
        _x_decoded_relu = merge([_deconv_3_decoded, decoder_deconv_4(_deconv_3_decoded)], mode="sum")


        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        #_x_decoded_mean = _x_decoded_mean_squash
        #_x_decoded_mean_np = Flatten()(_x_decoded_mean_squash)
        _x_decoded_mean = zp(_x_decoded_mean_squash)
        #_x_decoded_mean = Reshape((w, h, 1))(_x_decoded_mean)
        _x_decoded_mean = Flatten()(_x_decoded_mean)

        return decoder_input, x_decoded_mean, _x_decoded_mean
