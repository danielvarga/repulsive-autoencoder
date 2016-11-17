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


def build_model(batch_size, original_dim, dense_encoder, latent_dim, dense_decoder, nonvariational=False, spherical=False, convolutional=False):
    x = Input(batch_shape=(batch_size, original_dim))
    h = dense_encoder(x)
    if nonvariational:
        z = add_nonvariational(h, latent_dim)
        if spherical:
            z = Lambda(lambda z_unnormed: K.l2_normalize(z_unnormed, axis=-1))([z])
        encoder = Model(x, z)
        latent_layers = (z)
    else:
        assert not spherical, "Don't know how to normalize ellipsoids."

        z_mean, z_log_var = add_variational(h, latent_dim)
        encoder = Model(x, z_mean)
        latent_layers = (z_mean, z_log_var)
        z = add_sampling(z_mean, z_log_var, batch_size, latent_dim)

    decoder_input, x_decoded, _x_decoded = dense_decoder(z)

    # build autoencoder model
    vae = Model(x, x_decoded)
    # build a digit generator that can sample from the learned distribution
    generator = Model(decoder_input, _x_decoded)
    if not nonvariational: assert not spherical

    loss, metrics = loss_factory(nonvariational, spherical, convolutional, vae, original_dim, latent_layers)
    optimizer = RMSprop(lr=0.001)
    vae.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    return vae, encoder, generator


def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))

def spherical_sampler(batch_size, latent_dim):
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    z_sample /= np.linalg.norm(z_sample)
    return z_sample


def loss_factory(nonvariational, spherical, convolutional, model, original_dim, layers):
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
    def layerwise_loss(x, x_decoded):
        model_nodes = model.nodes_by_depth
        encOutputs = []
        decInputs = []
        for j in reversed(range(len(model_nodes))):
            node = model_nodes[j][0]
            outLayer = node.outbound_layer
            if outLayer.name.find("dec_conv") != -1:
                decInputs.append(node.input_tensors[0])
            if outLayer.name.find("enc_act") != -1:
                encOutputs.append(node.output_tensors[0])

        loss = 0
        for i in range(len(encOutputs)):
            encoder_output = encOutputs[i]
            decoder_input = decInputs[len(decInputs)-1-i]
            enc_shape = K.int_shape(encoder_output)[1:]
            dec_shape = K.int_shape(decoder_input)[1:]
            assert enc_shape == dec_shape, "encoder ({}) - decoder ({}) shape mismatch at layer {}".format(enc_shape, dec_shape, i)
            current_loss = original_dim * K.mean(K.batch_flatten(K.square(decoder_input - encoder_output)), axis=-1)
            loss += current_loss
        return 0.1 * K.mean(loss)

    metrics = [xent_loss]
    if not spherical:
        metrics.append(size_loss)
    if not nonvariational:
        metrics.append(variance_loss)
    if convolutional:
        metrics.append(layerwise_loss)
    
    def lossFun(x, x_decoded):
        loss = 0
        for metric in metrics:
            loss += metric(x, x_decoded)
        return loss
    return lossFun, metrics
