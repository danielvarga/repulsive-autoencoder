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


def build_model(batch_size, original_dim, intermediate_dims, latent_dim, nonvariational=False):
    x = Input(batch_shape=(batch_size, original_dim))
    h = x
    for intermediate_dim in intermediate_dims:
        h = Dense(intermediate_dim, activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    if not nonvariational:
        z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    if nonvariational:
        z = z_mean
    else:
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them both for reconstruction and generation
    decoder_input = Input(shape=(latent_dim,))
    h_decoded = z
    _h_decoded = decoder_input
    for intermediate_dim in reversed(intermediate_dims):
        decoder_h = Dense(intermediate_dim, activation='relu')
        h_decoded = decoder_h(h_decoded)    
        _h_decoded = decoder_h(_h_decoded)

    decoder_mean = Dense(original_dim, activation='sigmoid', name="decoder_mean")    
    x_decoded_mean = decoder_mean(h_decoded)
    _x_decoded_mean = decoder_mean(_h_decoded)

    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        if nonvariational:
            kl_loss = 0.5 * K.sum(K.square(z), axis=-1)
        else:
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae = Model(x, x_decoded_mean)
    optimizer = RMSprop(lr=0.001)
    vae.compile(optimizer=optimizer, loss=vae_loss)

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # build a digit generator that can sample from the learned distribution
    generator = Model(decoder_input, _x_decoded_mean)

    return vae, encoder, generator


def sample(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))
