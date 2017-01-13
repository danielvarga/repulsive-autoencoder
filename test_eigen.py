import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import eigen


batch_size = 256
latent_dim = 128


# Pushes latent datapoints in the direction of a randomly chosen hyperplane.
def random_vect_loss(z):
    v = K.random_normal_variable((latent_dim, 1), 0, 1)
    v = v / K.sqrt(K.dot(K.transpose(v), v))
    loss = K.square(K.dot(z, v))
    loss = K.mean(loss)
    return loss


# Pushes latent datapoints in the direction of the hyperplane that is
# orthogonal to the dominant eigenvector of the covariance matrix of the minibatch.
# Note: the eigenvector calculation assumes that the latent minibatch is zero-centered.
# We do not do this zero-centering.
def dominant_eigvect_loss(z):
    domineigvec, domineigval = eigen.eigvec(z, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
    loss = K.square(K.dot(z, domineigvec))
    loss = K.mean(loss)
    return loss


def test_loss():
    inputs = Input(shape=(latent_dim,))
    net = Dense(latent_dim)(inputs) # linear activation
    net = BatchNormalization()(net) # enforces a per-coordinate standard normal, to avoid collapsing into zero.

    # input's gonna be some ad hoc skewed normal,
    # output is expected to be standard normal,
    # the mechanism that enforces this is the dominant_eigvect_loss,
    # the method of verification is the eigenvalues of the covariance matrix of the output.

    # this is a tad too simplistic, what task could come after this, something a bit harder?

if __name__ == "__main__":
    test_loss()
