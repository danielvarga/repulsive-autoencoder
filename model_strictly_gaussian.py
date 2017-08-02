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

class StrictlyGaussianDecoder(Decoder):
    def __init__(self, args, generator_input):
        self.args = args
        self.main_channel = args.gaussianParams[0]
        self.dots = args.gaussianParams[1]

        self.gaussian_params = mixture.get_param_count(learn_variance, learn_density)
        self.main_params = self.main_channel * self.dots * self.gaussian_params
        self.side_params = args.gaussianParams[2]

        print "Main params: {}, Side params: {}".format(self.main_params, self.side_params)

    def __call__(self, recons_input):
        args = self.args
        generator_input = Input(batch_shape=(args.batch_size,args.latent_dim))
        generator_output = generator_input
        recons_output = recons_input

            
        layers = []
        layers.append(Reshape([self.main_channel, self.dots, self.gaussian_params]))
        layers.append(Activation("sigmoid"))
        layers.append(mixture.MixtureLayer(self.args.original_shape[0], self.args.original_shape[1], learn_variance=learn_variance, learn_density=learn_density, variance=args.gaussianVariance, maxpooling=args.gaussianMaxpooling, name="mixtureLayer"))

        for layer in layers:
            generator_output = layer(generator_output)
            recons_output = layer(recons_output)

        output_shape = K.int_shape(recons_output)[1:]
        assert output_shape == self.args.original_shape, "Expected shape {}, got shape {}".format(self.args.original_shape, output_shape)

        return generator_input, recons_output, generator_output
