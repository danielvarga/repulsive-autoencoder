import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input, BatchNormalization, Lambda
from keras.regularizers import l2
from keras import backend as K

import model

class Encoder(object):
    pass

class Decoder(object):
    pass

class LadderDenseEncoder(Encoder):
    def __init__(self, args):
        self.args = args
        assert args.latent_dim % len(args.intermediate_dims) == 0, "Latent dim has to be divisible by len(intermediate_dims)"
        self.latent_by_layer = args.latent_dim // len(args.intermediate_dims)

    def __call__(self, x):
        args = self.args
        h = Flatten()(x)
        zs = []
        z_means = []
        z_log_vars = []
        for intermediate_dim in args.intermediate_dims:
            h = Dense(intermediate_dim, kernel_regularizer=l2(args.encoder_wd))(h)
            h = Activation(args.activation)(h)
            z, z_mean, z_log_var = model.add_sampling(h, args.sampling, args.sampling_std, args.batch_size, self.latent_by_layer, args.encoder_wd)
            zs.append(z)
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
        self.z = Merge(mode='concat')(zs)
        self.z_mean = Merge(mode='concat')(z_means)
        self.z_log_var = Merge(mode='concat')(z_log_vars)
        return h

    def get_latent_code(self):
        return (self.z, self.z_mean, self.z_log_var)
    

class LadderDenseDecoder(Decoder):
    def __init__(self, args):
        self.args = args
        assert args.latent_dim % len(args.intermediate_dims) == 0, "Latent dim has to be divisible by len(intermediate_dims)"
        self.latent_by_layer = args.latent_dim // len(args.intermediate_dims)

    def __call__(self, recons_input):
        args = self.args
        generator_input = Input(shape=(args.latent_dim,))

        for i, intermediate_dim in enumerate(reversed(args.intermediate_dims)):
            slice_layer = Lambda(lambda x: x[:,i*self.latent_by_layer:(i+1)*self.latent_by_layer], output_shape=(self.latent_by_layer,))
            generator_input_portion = slice_layer(generator_input)
            recons_input_portion = slice_layer(recons_input)
            if i == 0:
                recons_output = recons_input_portion
                generator_output = generator_input_portion
            else:
                recons_output = Merge(mode='concat')([recons_output, recons_input_portion])
                generator_output = Merge(mode='concat')([generator_output, generator_input_portion])
                
            layer = Dense(intermediate_dim, kernel_regularizer=l2(args.decoder_wd))
            recons_output = layer(recons_output)
            recons_output = Activation(args.activation)(recons_output)
            generator_output = layer(generator_output)
            generator_output = Activation(args.activation)(generator_output)

        decoder_top = []
        decoder_top.append(Dense(np.prod(args.original_shape), activation='sigmoid', name="decoder_top", kernel_regularizer=l2(args.decoder_wd)))
        decoder_top.append(Reshape(args.original_shape))
        for layer in decoder_top:
            recons_output = layer(recons_output)
            generator_output = layer(generator_output)
                
        return generator_input, recons_output, generator_output
