from keras.models import Model
import mixture
import numpy as np
from keras.layers import Dense, Reshape, Input, Lambda, Convolution2D, Flatten, merge, Deconvolution2D, Activation, BatchNormalization
import keras.backend as K
from keras.regularizers import l1, l2
import keras

import net_blocks
keras.layers.MixtureLayer = mixture.MixtureLayer

learn_variance=True
learn_density=True
upscale = False

def get_latent_dim(gaussianParams):
    main_channel = gaussianParams[0]
    dots = gaussianParams[1]
    gaussian_params = mixture.get_param_count(learn_variance, learn_density)
    main_params = main_channel * dots * gaussian_params
    side_params = gaussianParams[2]
    latent_dim = main_params + side_params
    return latent_dim


class Decoder(object):
    pass

class GaussianDecoder(Decoder):
    def __init__(self, args, generator_input):
        self.args = args
        self.main_channel = args.gaussianParams[0]
        self.dots = args.gaussianParams[1]

        self.gaussian_params = mixture.get_param_count(learn_variance, learn_density)
        self.main_params = self.main_channel * self.dots * self.gaussian_params
        self.side_params = args.gaussianParams[2]

        print "Main params: {}, Side params: {}".format(self.main_params, self.side_params)

        self.side_channel = self.side_params
        self.channel = self.main_channel + self.side_channel
        self.depth = args.depth
        if upscale:
            assert args.original_shape[0] % (2 ** args.depth) == 0
            assert args.original_shape[1] % (2 ** args.depth) == 0
            factors = range(args.depth+1)
        else:
            factors = [0] * (args.depth+1)
        self.ys = []
        self.xs = []
        for i in factors:
            self.ys.append(args.original_shape[0] / (2 ** i))
            self.xs.append(args.original_shape[1] / (2 ** i))

    def __call__(self, recons_input):
        args = self.args
        generator_input = Input(batch_shape=(args.batch_size,args.latent_dim))
        generator_output = generator_input
        recons_output = recons_input

        # split side channels off
        position_latent_params = args.latent_dim - self.side_params
        def sideFun(x):
            x = x[:,position_latent_params:]
            x = K.expand_dims(x, dim=1)
            x = K.expand_dims(x, dim=2)
            x = K.repeat_elements(x, self.ys[self.depth], axis=1)
            x = K.repeat_elements(x, self.xs[self.depth], axis=2)
            return x

        if self.side_params > 0:
            assert self.side_params <= args.latent_dim
            sideSplitter = Lambda(sideFun, output_shape=(self.ys[args.depth], self.xs[args.depth], self.side_params), name="sideSplitter")
            recons_side = sideSplitter(recons_output)
            generator_side = sideSplitter(generator_output)

        if self.main_params > 0:
            mainSplitter = Lambda(lambda x: x[:,:position_latent_params], output_shape=(position_latent_params,), name="mainSplitter")
            recons_main = mainSplitter(recons_output)
            generator_main = mainSplitter(generator_output)

            # add intermediate dense layers
            layers = net_blocks.dense_block(reversed(args.intermediate_dims), args.decoder_wd, args.decoder_use_bn, args.activation)

            # the last layer ensures the feature size aligns with gaussianParams
            layers.append(Dense(self.main_params, W_regularizer=l2(args.decoder_wd)))
            layers.append(Reshape([self.main_channel, self.dots, self.gaussian_params]))
            layers.append(Activation("sigmoid"))
            layers.append(mixture.MixtureLayer(self.ys[args.depth], self.xs[args.depth], learn_variance=learn_variance, learn_density=learn_density, variance=args.gaussianVariance, maxpooling=args.gaussianMaxpooling, name="mixtureLayer"))

            for layer in layers:
                generator_main = layer(generator_main)
                recons_main = layer(recons_main)
         
        if self.side_params == 0:
            assert self.main_params > 0
            generator_output = generator_main
            recons_output = recons_main
        else:
            if self.main_params == 0:
                generator_output = generator_side
                recons_output = recons_side
            else:
                generator_output = merge([generator_main, generator_side], mode='concat', concat_axis=3)
                recons_output = merge([recons_main, recons_side], mode='concat', concat_axis=3)


        layers = []
        for i in reversed(range(args.depth)):
            layers.append(Convolution2D(self.channel, 3, 3, subsample=(1,1), border_mode="same"))
            if args.decoder_use_bn:
                layers.append(BatchNormalization())
            layers.append(Activation(args.activation))
            if upscale:
                layers.append(Deconvolution2D(self.channel, 2, 2, output_shape=(args.batch_size, self.ys[i], self.xs[i], self.channel), border_mode='same', subsample=(2,2), W_regularizer=l2(args.decoder_wd)))
                layers.append(Activation(args.activation))
        layers.append(Convolution2D(args.original_shape[2], 3, 3, activation="sigmoid", subsample=(1,1), border_mode="same"))

        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)
        
        output_shape = K.int_shape(recons_output)[1:]
        assert output_shape == self.args.original_shape, "Expected shape {}, got shape {}".format(self.args.original_shape, output_shape)

        # create a separate model that returns the output of the main channels
        # args.mixture_model = Model(generator_input, generator_main)
        # args.mixture_model.compile(optimizer="sgd", loss="mse")

        return generator_input, recons_output, generator_output

def add_noise(x, magnitude, batch_size):
    shape = [batch_size] + list(K.int_shape(x)[1:])
    noise = K.random_normal(shape=shape, mean=0., std=magnitude)
    return Lambda(lambda x: noise + x)((x))
    
