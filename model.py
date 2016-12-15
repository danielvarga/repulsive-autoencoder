'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

import dense
import model_conv_discgen

from keras.layers import Input, Dense, Lambda, Reshape, Flatten, Activation
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import *
import activations
from loss import loss_factory
from arm import ArmLayer

def build_model(args):
    x = Input(batch_shape=([args.batch_size] + list(args.original_shape)))

    if args.encoder == "dense":
        encoder = dense.DenseEncoder(args.intermediate_dims,args.activation)
    elif args.encoder == "conv":
        encoder = model_conv_discgen.ConvEncoder(
            depth = args.depth,
            latent_dim = args.latent_dim,
            intermediate_dims = args.intermediate_dims,
            image_dims = args.original_shape,
            batch_size = args.batch_size,
            base_filter_num = args.base_filter_num)
    hidden = encoder(x)

    z, z_mean, z_log_var = add_sampling(hidden, args.sampling, args.batch_size, args.latent_dim)

    if args.spherical:
        z = Lambda(lambda z_unnormed: K.l2_normalize(z_unnormed, axis=-1))([z])

    if args.decoder == "dense":
        decoder = dense.DenseDecoder(args.latent_dim, args.intermediate_dims, args.original_shape, args.activation)
    elif args.decoder == "conv":
        decoder = model_conv_discgen.ConvDecoder(
            depth = args.depth,
            latent_dim = args.latent_dim,
            intermediate_dims =args.intermediate_dims,
            image_dims = args.original_shape,
            batch_size = args.batch_size,
            base_filter_num = args.base_filter_num,
            wd = args.decoder_wd,
            use_bn = args.decoder_use_bn)
    generator_input, recons_output, generator_output = decoder(z)

    encoder = Model(x, z_mean)
    encoder_var = Model(x, z_log_var)
    ae = Model(x, recons_output)
    generator = Model(generator_input, generator_output)

    armLayer = ArmLayer(dict_size=1000, iteration=10, threshold=0.02, reconsCoef=1)
    sparse_input = Flatten()(x)
    sparse_input = armLayer(sparse_input)
    sparse_output = Flatten()(recons_output)
    sparse_output = armLayer(sparse_output)
    loss_features = (z, z_mean, z_log_var, sparse_input, sparse_output)
    loss, metrics = loss_factory(ae, encoder, loss_features, args)
    optimizer = RMSprop(lr=args.lr)
    ae.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return ae, encoder, encoder_var, generator


def add_sampling(hidden, sampling, batch_size, latent_dim):
    z_mean = Dense(latent_dim)(hidden)
    if not sampling:
        z_log_var = Lambda(lambda x: 0*x, output_shape=[latent_dim])((z_mean))
        return z_mean, z_mean, z_log_var
    else:
        z_log_var = Dense(latent_dim)(hidden)
        def sampling(inputs):
            z_mean, z_log_var = inputs
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        return z, z_mean, z_log_var


def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))

def spherical_sampler(batch_size, latent_dim):
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    z_sample /= np.linalg.norm(z_sample)
    return z_sample


