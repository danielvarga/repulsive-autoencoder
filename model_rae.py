'''
Repulsive autoencoder. This is an autoencoder that differs from a standard VAE in the following ways:

- The latent variances are not learned, they are simply set to 0.
- A normalization step takes the latent variables to a sphere surface.
- The regularization loss is changed to an energy term that
  corresponds to a pairwise repulsive force between the encoded
  elements of the minibatch.
'''

import numpy as np

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives


def build_model(batch_size, original_dim, intermediate_dim, latent_dim):
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)

    # Completely got rid of the variational aspect
    z_unnormalized = Dense(latent_dim)(h)

    # normalize all latent vars:
    z = Lambda(lambda z_unnormed: K.l2_normalize(z_unnormed, axis=-1))([z_unnormalized])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded = decoder(h_decoded)


    def vae_loss(x, x_decoded):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
        # Instead of a KL normality test, here's some energy function
        # that pushes the minibatch elements away from each other, pairwise.
        # pairwise = K.sum(K.square(K.dot(z, K.transpose(z))))

        epsilon = 0.0001
        distances = (2.0 + epsilon - 2.0 * K.dot(z, K.transpose(z))) ** 0.5
        regularization = -K.mean(distances) * 100 # Keleti
        return xent_loss + regularization

    rae = Model(x, x_decoded)
    rae.compile(optimizer='rmsprop', loss=vae_loss)

    # build a model to project inputs on the latent space
    encoder = Model(x, z)

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded = decoder(_h_decoded)
    generator = Model(decoder_input, _x_decoded)

    return rae, encoder, generator


# Taken uniformly from sphere in R^latentdim
def sample(batch_size, latent_dim):
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    z_sample /= np.linalg.norm(z_sample)
    return z_sample
