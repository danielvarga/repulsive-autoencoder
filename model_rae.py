import numpy as np

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives



def build_rae(batch_size, original_dim, intermediate_dim, latent_dim):
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
        regularization = -K.mean(distances) * 1000 # Keleti
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
