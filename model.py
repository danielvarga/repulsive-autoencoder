'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.optimizers import *


class Encoder(object):
    pass

class DenseEncoder(Encoder):
    def __init__(self, intermediate_dims):
        self.intermediate_dims = intermediate_dims

    def __call__(self, x):
        h = x
        for intermediate_dim in self.intermediate_dims:
            h = Dense(intermediate_dim, activation='relu')(h)
        return h


def add_variational(h, latent_dim):
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    return z_mean, z_log_var

def add_sampling(z_mean, z_log_var, batch_size, latent_dim):
    def sampling(inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    return z

def add_nonvariational(h, latent_dim):
    z = Dense(latent_dim)(h)
    return z


class Decoder(object):
    pass # TODO interface

class DenseDecoder(Decoder):
    def __init__(self, latent_dim, intermediate_dims, original_dim):
        self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.original_dim = original_dim

    def __call__(self, z):
        # we instantiate these layers separately so as to reuse them both for reconstruction and generation
        decoder_input = Input(shape=(self.latent_dim,))
        h_decoded = z
        _h_decoded = decoder_input
        for intermediate_dim in reversed(self.intermediate_dims):
            decoder_h = Dense(intermediate_dim, activation='relu')
            h_decoded = decoder_h(h_decoded)    
            _h_decoded = decoder_h(_h_decoded)
        decoder_top = Dense(self.original_dim, activation='sigmoid', name="decoder_top")
        x_decoded = decoder_top(h_decoded)
        _x_decoded = decoder_top(_h_decoded)
        return decoder_input, x_decoded, _x_decoded


def build_model(batch_size, original_dim, dense_encoder, latent_dim, dense_decoder, nonvariational=False, spherical=False):
    x = Input(batch_shape=(batch_size, original_dim))
    h = dense_encoder(x)
    if nonvariational:
        z = add_nonvariational(h, latent_dim)
    else:
        z_mean, z_log_var = add_variational(h, latent_dim)
        z = add_sampling(z_mean, z_log_var, batch_size, latent_dim)

    if spherical:
        assert nonvariational, "Don't know how to normalize ellipsoids."
        z = Lambda(lambda z_unnormed: K.l2_normalize(z_unnormed, axis=-1))([z])

    decoder_input, x_decoded, _x_decoded = dense_decoder(z)

    vae = Model(x, x_decoded)
    if nonvariational:
        if spherical:
            loss, metrics = loss_factory("rae", original_dim, (z))
        else:
            loss, metrics = loss_factory("nvae", original_dim, (z))
    else:
        assert not spherical
        loss, metrics = loss_factory("vae", original_dim, (z_mean, z_log_var))

    optimizer = RMSprop(lr=0.001)
    vae.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # build a model to project inputs on the latent space
    if nonvariational:
        encoder = Model(x, z)
    else:
        encoder = Model(x, z_mean)

    # build a digit generator that can sample from the learned distribution
    generator = Model(decoder_input, _x_decoded)

    return vae, encoder, generator


def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))

def spherical_sampler(batch_size, latent_dim):
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    z_sample /= np.linalg.norm(z_sample)
    return z_sample


def loss_factory(model_type, original_dim, layers):
    def xent_loss(x, x_decoded):
        loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)
    def mse_loss(x, x_decoded):
        loss = original_dim * objectives.mean_squared_error(x, x_decoded)
        return K.mean(loss)
    def size_loss(x, x_decoded):
        loss = 0.5 * K.sum(K.square(layers[0]), axis=-1)
        return K.mean(loss)
    def variance_loss(x, x_decoded):
        loss = 0.5 * K.sum(-1 - layers[1] + K.exp(layers[1]), axis=-1)
        return K.mean(loss)

    if (model_type == "rae"):
        metrics = [xent_loss]
    elif (model_type == "nvae"):
        metrics = [xent_loss, size_loss]
    elif (model_type == "vae"):
        metrics = [xent_loss, size_loss, variance_loss]
    else:
        assert False, "loss for model type $s not yet implemented" % model_type
    
    def lossFun(x, x_decoded):
        loss = 0
        for metric in metrics:
            loss += metric(x, x_decoded)
        return loss
    return lossFun, metrics
