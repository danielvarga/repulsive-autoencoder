'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

from keras.layers import Input, Dense, Lambda, Convolution2D, Deconvolution2D, Reshape, Flatten, ZeroPadding2D, merge, Activation, Layer
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import *
from keras.regularizers import l1, l2





class Unpooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        super(Unpooling2D,self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize,
            "ignore_border":self.ignore_border}



add_tables=[]



def residual_drop(x, input_shape, output_shape, strides=(1, 1), weight_decay=0.003):
    global add_tables

    if K.image_dim_ordering() == 'th':
        feat_axis = 1
    else:
        feat_axis = 3


    nb_filter = output_shape[0]
    conv = Convolution2D(nb_filter, (3, 3), strides=strides,
                         padding="same", kernel_regularizer=l2(weight_decay))(x)
    conv = BatchNormalization(axis=feat_axis)(conv)
    conv = Activation("relu")(conv)
    conv = Convolution2D(nb_filter, (3, 3),
                         padding="same", kernel_regularizer=l2(weight_decay))(conv)
    conv = BatchNormalization(axis=feat_axis)(conv)

    if strides[0] >= 2:
        x = AveragePooling2D(strides)(x)

    if (output_shape[0] - input_shape[0]) > 0:
        pad_shape = (1,
                     output_shape[0] - input_shape[0],
                     output_shape[1],
                     output_shape[2])
        padding = K.zeros(pad_shape)
        padding = K.repeat_elements(padding, K.shape(x)[0], axis=0)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=1),
                   output_shape=output_shape)(x)

    death_rate = 0
    _death_rate = K.variable(death_rate)
    scale = K.ones_like(conv) - _death_rate
    conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
                  output_shape=output_shape)(conv)

    out = merge([conv, x], mode="sum")
    out = Activation("relu")(out)

    gate = K.variable(1, dtype="uint8")
    add_tables += [{"death_rate": _death_rate, "gate": gate}]
    return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                  output_shape=output_shape)([out, x])

def residual_drop_deconv(x, input_shape, output_shape, strides=(1, 1), weight_decay=0.003):
    global add_tables

    if K.image_dim_ordering() == 'th':
        feat_axis = 1
    else:
        feat_axis = 3


    nb_filter = output_shape[0]
    nb_conv = 3

    conv = Convolution2D(nb_filter, (3, 3), strides=strides,
                         padding="same", kernel_regularizer=l2(weight_decay))(x)
    decoder_deconv_1 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=strides,
                                           activation='linear',
                                           kernel_regularizer=l2(weight_decay))
 
    conv = BatchNormalization(axis=feat_axis)(conv)
    conv = Activation("relu")(conv)
    decoder_deconv_1 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=strides,
                                           activation='linear',
                                           kernel_regularizer=l2(weight_decay))
 
    conv = BatchNormalization(axis=feat_axis)(conv)
    conv = Activation("relu")(conv)
 
    if strides[0] >= 2:
        x = Unpooling2D(strides)(x)

    if (output_shape[0] - input_shape[0]) > 0:
        pad_shape = (1,
                     output_shape[0] - input_shape[0],
                     output_shape[1],
                     output_shape[2])
        padding = K.zeros(pad_shape)
        padding = K.repeat_elements(padding, K.shape(x)[0], axis=0)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=1),
                   output_shape=output_shape)(x)

    death_rate = 0
    _death_rate = K.variable(death_rate)
    scale = K.ones_like(conv) - _death_rate
    conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
                  output_shape=output_shape)(conv)

    out = merge([conv, x], mode="sum")
    out = Activation("relu")(out)

    gate = K.variable(1, dtype="uint8")
    add_tables += [{"death_rate": _death_rate, "gate": gate}]
    return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                  output_shape=output_shape)([out, x])

class Encoder(object):
    pass

class ConvEncoder(Encoder):
    def __init__(self, levels_config, filter_num_config, latent_dim, img_size, activation_config=None, batch_size=32, wd=0.003):
        self.levels_config = levels_config
        self.filter_num_config = filter_num_config
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.wd = wd
        self.filter_size = 3

    def __call__(self, x):

        image_chns = 1
        if K.image_dim_ordering() == 'th':
            feat_axis = 1
        else:
            feat_axis = 3

        x_reshaped = Reshape(self.img_size)(x)

        filter_num_config = self.filter_num_config

        net = Convolution2D(filter_num_config[0], (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x_reshaped)
        net = BatchNormalization(axis=feat_axis)(net)
        net = Activation("relu")(net)

        N = self.levels_config[0]

        for i in range(N):
            net = residual_drop(net, input_shape=(filter_num_config[0], self.img_size[0], self.img_size[1]), output_shape=(filter_num_config[0], self.img_size[0], self.img_size[1]))

        net = residual_drop(
            net,
            input_shape=(filter_num_config[0], self.img_size[0], self.img_size[1]),
            output_shape=(filter_num_config[1], self.img_size[0]//2, self.img_size[1]//2),
            strides=(2, 2)
        )
        for i in range(N - 1):
            net = residual_drop(
                net,
                input_shape=(filter_num_config[1], self.img_size[0]//2, self.img_size[1]//2),
                output_shape=(filter_num_config[1], self.img_size[0]//2, self.img_size[1]//2)
            )

        net = residual_drop(
            net,
            input_shape=(filter_num_config[1], self.img_size[0]//2, self.img_size[1]//2),
            output_shape=(filter_num_config[2], self.img_size[0]//4, self.img_size[1]//4),
            strides=(2, 2)
        )
        for i in range(N - 1):
            net = residual_drop(
                net,
                input_shape=(filter_num_config[2], self.img_size[0]//4, self.img_size[1]//4),
                output_shape=(filter_num_config[2], self.img_size[0]//4, self.img_size[1]//4)
            )

        pool = AveragePooling2D((8, 8))(net)
        flatten = Flatten()(pool)

        z = Dense(self.latent_dim, activation="softmax", W_regularizer=l2(weight_decay))(flatten)

        return z


class Decoder(object):
    pass # TODO interface

class ConvDecoder(Decoder):

    def __init__(self, latent_dim, intermediate_dim, original_dim, img_size, batch_size=32, wd=0.003):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.wd = wd
        self.filter_size = 3

    def __call__(self, z):

        img_chns = 1
        nb_conv = 3
        batch_size = self.batch_size
        nb_filters = self.intermediate_dim

        if K.image_dim_ordering() == 'th':
            feat_axis = 1
        else:
            feat_axis = 3

        h = self.img_size[0]
        w = self.img_size[1]

        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        decoder_input = Input(shape=(self.latent_dim,))


        h_decoded = z
        _h_decoded = decoder_input

        decoder_hid = Dense(self.intermediate_dim)


        decoder_upsample2 = Dense(4*(h//2) * (w//2), W_regularizer=l2(self.wd))



        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, h//2, w//2)
        else:
            output_shape = (batch_size, h//2, w//2, nb_filters)
    
        decoder_reshape = Reshape(output_shape[1:])

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, h//2, w//2)
        else:
            output_shape = (batch_size, h//2, w//2, nb_filters)

        decoder_deconv_1 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                       padding='same',
                                       strides=(1, 1),
                                       activation='relu',
                                       kernel_regularizer=l2(self.wd))

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, 4, h//2, w//2)
            feat_axis = 1
        else:
            output_shape = (batch_size, h//2, w//2, 4)
            feat_axis = 1



        decoder_reshape = Reshape(output_shape[1:])

        if K.image_dim_ordering() == 'th':
            output_shape = (batch_size, nb_filters, h//2, w//2)
        else:
            output_shape = (batch_size, h//2, w//2, nb_filters)

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
                                                  activation='relu', name="upsz", kernel_regularizer=l2(self.wd))

        decoder_deconv_4 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=(1, 1),
                                           activation='relu')

        decoder_mean_squash = Convolution2D(img_chns, (2, 2),
                                            padding='same',
                                            activation='sigmoid', name='reconv_flatten', kernel_regularizer=l2(self.wd))

        zp = ZeroPadding2D({'top_pad':1, 'left_pad':1, 'right_pad':0, 'bottom_pad':0})


        hid_decoded = decoder_hid(z)
        #up_decoded = decoder_upsample(hid_decoded)
        up_decoded = decoder_upsample2(hid_decoded)
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
        #_up_decoded = decoder_upsample(_hid_decoded)
        _up_decoded = decoder_upsample2(_hid_decoded)
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
