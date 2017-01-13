import mixture
import numpy as np
from keras.layers import Dense, Reshape, Input, Lambda, Convolution2D, Flatten, merge, Deconvolution2D, Activation
import keras.backend as K
from keras.regularizers import l1, l2
import keras
keras.layers.MixtureLayer = mixture.MixtureLayer

class Decoder(object):
    pass

class GaussianDecoder(Decoder):
    def __init__(self, args):
        self.args = args
        self.main_channel = args.gaussianParams[0]
        self.dots = args.gaussianParams[1]

        self.gaussian_params = mixture.GAUSS_PARAM_COUNT
        self.main_params = self.main_channel * self.dots * self.gaussian_params
        assert self.main_params < args.latent_dim
        self.side_params = args.latent_dim - self.main_params

        self.side_channel = args.latent_dim
        self.channel = self.main_channel + self.side_channel
        self.depth = args.depth        
        assert args.original_shape[0] % (2 ** args.depth) == 0
        assert args.original_shape[1] % (2 ** args.depth) == 0
        self.ys = []
        self.xs = []
        for i in range(args.depth+1):
            self.ys.append(args.original_shape[0] / (2 ** i))
            self.xs.append(args.original_shape[1] / (2 ** i))
        self.learn_variance=True # TODO
        self.variance = 1

    def __call__(self, recons_input):
        args = self.args

        generator_input = Input(shape=(args.latent_dim,))
        generator_output = generator_input
        recons_output = recons_input

        mainSplitter = Lambda(lambda x: x[:,:self.main_params], output_shape=(self.main_params,))
        generator_main = mainSplitter(generator_output)
        recons_main = mainSplitter(recons_output)

        def sideFun(x):
            x = x[:,self.main_params:]
            x = K.expand_dims(x, dim=1)
            x = K.expand_dims(x, dim=2)
            x = K.repeat_elements(x, self.ys[self.depth], axis=1)
            x = K.repeat_elements(x, self.xs[self.depth], axis=2)
            return x

        sideSplitter = Lambda(sideFun, output_shape=(self.ys[args.depth], self.xs[args.depth], self.side_params,))
        generator_side = sideSplitter(generator_output)
        recons_side = sideSplitter(recons_output)

        mainLayers = []
        mainLayers.append(Reshape([self.main_channel, self.dots, self.gaussian_params]))
        mainLayers.append(mixture.MixtureLayer(self.ys[args.depth], self.xs[args.depth], self.main_channel, learn_variance=self.learn_variance, variance=self.variance, maxpooling=False))
        for layer in mainLayers:
            generator_main = layer(generator_main)
            recons_main = layer(recons_main)

        generator_output = merge([generator_main, generator_side], mode='concat', concat_axis=3)
        recons_output = merge([recons_main, recons_side], mode='concat', concat_axis=3)


        layers = []
        for i in reversed(range(args.depth)):
            layers.append(Convolution2D(self.channel, 3, 3, subsample=(1,1), border_mode="same"))
            layers.append(Activation(args.activation))
            layers.append(Deconvolution2D(self.channel, 2, 2, output_shape=(args.batch_size, self.ys[i], self.xs[i], self.channel), border_mode='same', subsample=(2,2), W_regularizer=l2(args.decoder_wd)))
            layers.append(Activation(args.activation))
        layers.append(Convolution2D(args.original_shape[2], 3, 3, activation="sigmoid", subsample=(1,1), border_mode="same"))

        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)
        
        output_shape = K.int_shape(recons_output)[1:]
        assert output_shape == self.args.original_shape, "Expected shape {}, got shape {}".format(self.args.original_shape, output_shape)

        return generator_input, recons_output, generator_output 
