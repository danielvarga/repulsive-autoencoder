import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import eigen


batch_size = 256
latent_dim = 32


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


def cholesky(d):
    cov = np.cov(d.T)
    mean = np.mean(d, axis=0)

    print "means =", mean
    print "deviations = ", np.std(d, axis=0)

    eigVals, eigVects = np.linalg.eigh(cov)
    print "eigvals =", sorted(eigVals, reverse=True)

    cho = np.linalg.cholesky(cov)

    N = 10000
    z = np.random.normal(0.0, 1.0, (N, latent_dim))
    simulated = cho.dot(z.T).T + mean



def test_loss():
    inputs = Input(shape=(latent_dim,))
    net = Dense(latent_dim)(inputs) # linear activation
    z = BatchNormalization()(net) # enforces a per-coordinate standard normal, to avoid collapsing into zero.

    loss = lambda x, x_pred: dominant_eigvect_loss(z)
    # loss = lambda x, x_pred: random_vect_loss(z)

    model = Model(input=inputs, output=z)
    optimizer = Adam(lr=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=loss)

    N = 10000
    megaepoch_count = 10

    for i in range(megaepoch_count):
        data = np.random.normal(size=(N, latent_dim))
        data[:, 0] += data[:, 1]

        model.fit(data, data, batch_size=batch_size, verbose=2)

        output = model.predict(data)

        cholesky(output)


if __name__ == "__main__":
    test_loss()
