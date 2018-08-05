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

class Decoder(object):
    pass # TODO interface

class ResnetDecoder(Decoder):

    def __init__(self, args):
        self.args = args

    def __call__(self, recons_latent):
        generator_latent = Input(batch_shape=(self.args.batch_size, self.args.latent_dim), name="generator_latent")

        # initial outputs
        recons_output = recons_latent
        generator_output = generator_latent
        layers = []
        layers.append(Dense(np.prod(self.args.original_shape), name="initial_output"))
        layers.append(Activation(self.args.activation))
        layers.append(Reshape(self.args.original_shape))
        for layer in layers:
            recons_output = layer(recons_output)
            generator_output = layer(generator_output)

        intermediary_outputs = []
        for i in range(self.args.depth):
            recons_output, generator_output = resnet_block(recons_output, generator_output, recons_latent, generator_latent, self.args)
            intermediary_outputs.append(recons_output)

        layers = []
        layers.append(Convolution2D(self.args.original_shape[2], 3, 3, activation="sigmoid", subsample=(1,1), border_mode="same"))
        for layer in layers:
            recons_output = layer(recons_output)
            generator_output = layer(generator_output)

        return generator_latent, recons_output, generator_output, intermediary_outputs


def resnet_block(input1, input2, latent1, latent2, args):
    input_shape = K.int_shape(input1)[1:]
    input2_shape = K.int_shape(input2)[1:]
    assert input_shape == input2_shape

    # turn the input with shape (batch_size, params) into (batch_size, sizeX, sizeY, params), where each channel is constant
    def dense2Conv(input):
        input = K.expand_dims(input, dim=1)
        input = K.expand_dims(input, dim=2)
        input = K.repeat_elements(input, input_shape[0], axis=1)
        input = K.repeat_elements(input, input_shape[1], axis=2)
        return input
    
    # apply random projection on the latent code into block_params values and create block_params layers out of them
    block_params = 5
    layers = []
    layers.append(Dense(block_params, trainable=False))            
    layers.append(Lambda(dense2Conv))
    for layer in layers:
        latent1 = layer(latent1)
        latent2 = layer(latent2)

    # merge the latent channel into the output
    output1 = merge([input1, latent1], mode="concat", concat_axis=3)
    output2 = merge([input2, latent2], mode="concat", concat_axis=3)

    layers = []
    layers.append(Convolution2D(block_params + input_shape[2], 3, 3, subsample=(1,1), border_mode="same"))
    layers.append(BatchNormalization())
    layers.append(Activation(args.activation))
    layers.append(Convolution2D(block_params + input_shape[2], 3, 3, subsample=(1,1), border_mode="same"))
    layers.append(BatchNormalization())
    layers.append(Activation(args.activation))
    layers.append(Convolution2D(input_shape[2], 3, 3, subsample=(1,1), border_mode="same"))
    layers.append(BatchNormalization())
    for layer in layers:
        output1 = layer(output1)
        output2 = layer(output2)

    result1 = merge([input1, output1], mode="sum")
    result2 = merge([input2, output2], mode="sum")
    result1 = Activation(args.activation)(result1)
    result2 = Activation(args.activation)(result2)

    return result1, result2
