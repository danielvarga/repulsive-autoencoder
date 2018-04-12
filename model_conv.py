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

class Unpooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(Unpooling2D, self).__init__()
        #self.input = T.tensor4()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0],
                self.poolsize[0] * input_shape[1],
                self.poolsize[1] * input_shape[2],  input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=1).repeat(s2, axis=2)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}


add_tables=[]
decoder_layers = []

def residual_drop(x, input_shape, output_shape, strides=(1, 1), weight_decay=0.003, use_batchnorm=False):
    global add_tables

    if K.image_dim_ordering() == 'th':
        feat_axis = 1
    else:
        feat_axis = 3


    nb_filter = output_shape[3]
    conv = Convolution2D(nb_filter, (3, 3), strides=strides,
                         padding="same", kernel_regularizer=l2(weight_decay))(x)
    if use_batchnorm:
        conv = BatchNormalization(axis=feat_axis)(conv)
    conv = Activation("relu")(conv)
    conv = Convolution2D(nb_filter, (3, 3),
                         padding="same", kernel_regularizer=l2(weight_decay))(conv)

    if use_batchnorm:
        conv = BatchNormalization(axis=feat_axis)(conv)

    if strides[0] >= 2:
        x = AveragePooling2D(strides)(x)


    if (output_shape[3] - input_shape[3]) > 0:
        pad_shape = (1,
                     output_shape[1],
                     output_shape[2],
                     output_shape[3] - input_shape[3])
        padding = K.zeros(pad_shape)
        print(output_shape)
        padding = K.repeat_elements(padding, output_shape[0], axis=0)

        print(padding)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=3),
                   output_shape=output_shape[1:])(x)
    
    #death_rate = 0
    #_death_rate = K.variable(death_rate)
    #scale = K.ones_like(conv) - _death_rate
    #conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
    #              output_shape=output_shape)(conv)

    out = merge([conv, x], mode="sum")
    out = Activation("relu")(out)
    return out

    #gate = K.variable(1, dtype="uint8")
    #add_tables += [{"death_rate": _death_rate, "gate": gate}]
    #return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
    #              output_shape=output_shape)([out, x])

def residual_drop_deconv(x, input_shape, output_shape, strides=(1, 1), weight_decay=0.003, act="relu", with_batchnorm=True):
    global add_tables

    if K.image_dim_ordering() == 'th':
        feat_axis = 1
    else:
        feat_axis = 3


    nb_filters = output_shape[3]
    nb_conv = 3

    deconv_1 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=strides,
                                           activation='linear',
                                           kernel_regularizer=l2(weight_decay))
    decoder_layers.append(deconv_1)

    bn_1 = BatchNormalization(axis=feat_axis)
    if with_batchnorm:
        decoder_layers.append(bn_1)

    act_1 = Activation(act)
    decoder_layers.append(act_1)

    deconv_2 = Deconvolution2D(nb_filters, (nb_conv, nb_conv),
                                           padding='same',
                                           strides=(1,1),
                                           activation='linear',
                                           kernel_regularizer=l2(weight_decay))
 
    decoder_layers.append(deconv_2)
    bn_2 = BatchNormalization(axis=feat_axis)
    if with_batchnorm:
        decoder_layers.append(bn_2)
    act_2 = Activation(act)
    decoder_layers.append(act_2)


    if strides[0] >= 2:
        #unpooling_1 = Unpooling2D(strides)
        #decoder_layers.append(unpooling_1)
        pass
 

    """
    if (output_shape[3] - input_shape[3]) > 0:
        pad_shape = (1,
                     output_shape[1],
                     output_shape[2],
                     output_shape[3] - input_shape[3])
        padding = K.zeros(pad_shape)
        print(output_shape)
        padding = K.repeat_elements(padding, output_shape[0], axis=0)
        print(padding)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=3),
                   output_shape=output_shape)
        decoder_layers.append(x)
    """
    """
    if (output_shape[0] - input_shape[0]) > 0:
        pad_shape = (1,
                     output_shape[0] - input_shape[0],
                     output_shape[1],
                     output_shape[2])
        padding = K.zeros(pad_shape)
        padding = K.repeat_elements(padding, K.shape(x)[0], axis=0)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=1),
                   output_shape=output_shape)(x)
    """

    #death_rate = 0
    #_death_rate = K.variable(death_rate)
    #scale = K.ones_like(conv) - _death_rate
    #conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
    #              output_shape=output_shape)(conv)

    #merge_1 = Merge(mode="sum")([conv, x])
    #decoder_layers.append(merge_1)

    return None


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

        net = Convolution2D(filter_num_config[0], (3, 3), padding="same", kernel_regularizer=l2(self.wd))(x_reshaped)
        net = BatchNormalization(axis=feat_axis)(net)
        net = Activation("relu")(net)

        N = self.levels_config[0]

        for i in range(N):
            net = residual_drop(net, input_shape=(self.batch_size, self.img_size[0], self.img_size[1], filter_num_config[0]), output_shape=(self.batch_size, self.img_size[0], self.img_size[1], filter_num_config[0]))

        net = residual_drop(
            net,
            input_shape=(self.batch_size, self.img_size[0], self.img_size[1], filter_num_config[0]),
            output_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
            strides=(2, 2)
        )
        for i in range(N - 1):
            net = residual_drop(
                net,
                input_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
                output_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1])
            )

        net = residual_drop(
            net,
            input_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
            output_shape=(self.batch_size, self.img_size[0]//4, self.img_size[1]//4, filter_num_config[2]),
            strides=(2, 2)
        )
        for i in range(N - 1):
            net = residual_drop(
                net,
                input_shape=(self.batch_size, self.img_size[0]//4, self.img_size[1]//4, filter_num_config[2]),
                output_shape=(self.batch_size, self.img_size[0]//4, self.img_size[1]//4, filter_num_config[2])
            )

        #pool = AveragePooling2D((8, 8))(net)
        flatten = Flatten()(net)

        z = Dense(self.latent_dim, activation="softmax", W_regularizer=l2(self.wd))(flatten)

        return z


class Decoder(object):
    pass # TODO interface

class ConvDecoder(Decoder):

    def __init__(self, levels_config, filter_num_config, latent_dim, img_size, activation_config=None, batch_size=32, wd=0.003):
        self.levels_config = levels_config
        self.filter_num_config = filter_num_config
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.batch_size = batch_size
        self.wd = wd
        self.filter_size = 3

    def __call__(self, z):

        filter_num_config = self.filter_num_config
        img_chns = 1

        if K.image_dim_ordering() == 'th':
            feat_axis = 1
        else:
            feat_axis = 3

        h = self.img_size[0]
        w = self.img_size[1]

        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        decoder_input = Input(batch_shape=(self.batch_size, self.latent_dim,))


        h_decoded = z
        _h_decoded = decoder_input


        #decoder_layers.append(z)


        decoder_hid = Dense(self.filter_num_config[2])
        #decoder_layers.append(decoder_hid)

        #decoder_unpooling = Unpooling2D((18,15))
        #decoder_layers.append(decoder_unpooling)

        decoder_unpoolshit = Dense(self.filter_num_config[2]//4*18*15)
        decoder_layers.append(decoder_unpoolshit)


        #reshape_1 = Reshape((self.img_size[0]//4, self.img_size[1]//4, self.filter_num_config[2]))

        reshape_1 = Reshape((18,15, self.filter_num_config[2]//4))
        decoder_layers.append(reshape_1)

        net = None

        #decoder_layers.append(Flatten())


    
        
        N = self.levels_config[0]
        for i in range(N - 1):
            net = residual_drop_deconv(
                net,
                output_shape=(self.batch_size, self.img_size[0]//4, self.img_size[1]//4, filter_num_config[2]),
                input_shape=(self.batch_size, self.img_size[0]//4, self.img_size[1]//4, filter_num_config[2]),
                act="relu",
                with_batchnorm = False

            )
        
        net = residual_drop_deconv(
            net,

            output_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
            input_shape=(self.batch_size, self.img_size[0]//4, self.img_size[1]//4, filter_num_config[2]),
            strides=(2, 2),
            act="relu",
            with_batchnorm = False
        )

        
        for i in range(N - 1):
            net = residual_drop_deconv(
                net,
                output_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
                input_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
                act="relu",
                with_batchnorm = False

            )
        
        net = residual_drop_deconv(
            net,
            output_shape=(self.batch_size, self.img_size[0], self.img_size[1], filter_num_config[0]),
            input_shape=(self.batch_size, self.img_size[0]//2, self.img_size[1]//2, filter_num_config[1]),
            strides=(2, 2),
            act="tanh",
            with_batchnorm = False
        )
        

        for i in range(N):
            net = residual_drop_deconv(
                    net, 
                    output_shape=(self.batch_size, self.img_size[0], self.img_size[1], filter_num_config[0]), 
                    input_shape=(self.batch_size, self.img_size[0], self.img_size[1], filter_num_config[0]),
                    act="tanh",
                    with_batchnorm = False
            )

        
        conv_f = Convolution2D(1, (5, 5), strides=(1,1), padding="same")
        decoder_layers.append(conv_f)
        
        """
        bn_f = BatchNormalization(axis=feat_axis)
        decoder_layers.append(bn_f)

        act_f = Activation("tanh")
        decoder_layers.append(act_f)
        """

        flatten_f = Flatten()
        decoder_layers.append(flatten_f)

        """
        dense_f = Dense(h*w)
        decoder_layers.append(dense_f)
        """
        """
        re_f = Reshape((self.batch_size,h,w,1))
        decoder_layers.append(re_f)
        """

        x = decoder_input
        for layer in decoder_layers:
            print(K.shape(x))
            x = layer(x)
        _x_decoded_mean = x
            
        x = z
        for layer in decoder_layers:
            x = layer(x)
        x_decoded_mean = x
 
        return decoder_input, x_decoded_mean, _x_decoded_mean
