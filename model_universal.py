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


def add_encoder(x, intermediate_dims):
    h = x
    for intermediate_dim in intermediate_dims:
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


def add_decoder(z, latent_dim, intermediate_dims, original_dim):
    # we instantiate these layers separately so as to reuse them both for reconstruction and generation
    decoder_input = Input(shape=(latent_dim,))
    h_decoded = z
    _h_decoded = decoder_input
    for intermediate_dim in reversed(intermediate_dims):
        decoder_h = Dense(intermediate_dim, activation='relu')
        h_decoded = decoder_h(h_decoded)    
        _h_decoded = decoder_h(_h_decoded)
    decoder_top = Dense(original_dim, activation='sigmoid', name="decoder_top")
    x_decoded = decoder_top(h_decoded)
    _x_decoded = decoder_top(_h_decoded)
    return decoder_input, x_decoded, _x_decoded


def loss_factory(original_dim, z_mean, z_log_var):
    def vae_loss(x, x_decoded):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
    return vae_loss

def build_model(batch_size, original_dim, intermediate_dims, latent_dim):
    x = Input(batch_shape=(batch_size, original_dim))
    h = add_encoder(x, intermediate_dims)
    z_mean, z_log_var = add_variational(h, latent_dim)
    z = add_sampling(z_mean, z_log_var, batch_size, latent_dim)
    decoder_input, x_decoded, _x_decoded = add_decoder(z, latent_dim, intermediate_dims, original_dim)

    vae = Model(x, x_decoded)
    loss = loss_factory(original_dim, z_mean, z_log_var)
    optimizer = RMSprop(lr=0.001)
    vae.compile(optimizer=optimizer, loss=loss)

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # build a digit generator that can sample from the learned distribution
    generator = Model(decoder_input, _x_decoded)

    return vae, encoder, generator


def sample(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))
