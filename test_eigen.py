import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import eigen


batch_size = 256
latent_dim = 3
L2_REG_WEIGHT = 0.02

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


def l2_loss(z):
    return L2_REG_WEIGHT * K.mean(K.square(z)) * latent_dim # sum over latent_dim, avg over minibatch


def dominant_eigvect_layer(z):
    domineigvec, domineigval = eigen.eigvec(z, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
    domineigvec_stacked = K.repeat_elements(K.reshape(domineigvec, (1, latent_dim)), batch_size, axis=0)
    # print tuple(map(int, domineigvec_stacked.get_shape()))
    return domineigvec_stacked


def cholesky(d):
    cov = np.cov(d.T)
    mean = np.mean(d, axis=0)

    print "means =", mean
    print "deviations = ", np.std(d, axis=0)
    print d[:10]

    eigVals, eigVects = np.linalg.eigh(cov)
    print "eigvals =", list(reversed(eigVals))
    print "dominant eigvect =", eigVects[:, -1]

    cho = np.linalg.cholesky(cov)
    print "cholesky =", cho

    N = 10000
    z = np.random.normal(0.0, 1.0, (N, latent_dim))
    simulated = cho.dot(z.T).T + mean


def test_eigen():
    inputs = Input(shape=(latent_dim,))
    eigvec = Lambda(lambda z: dominant_eigvect_layer(z))([inputs])
    model = Model(input=inputs, output=eigvec)
    optimizer = Adam(lr=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=lambda x, x_pred: K.zeros((1, )))

    N = 512
    d = np.random.normal(size=(N, latent_dim))
    d[:, 0] += d[:, 1]
    cov = np.cov(d.T)
    print "cov", cov
    print "---"
    eigVals, eigVects = np.linalg.eigh(cov)
    for eigVal, eigVect in zip(eigVals, eigVects.T):
        print eigVal, "*", eigVect, "=", cov.dot(eigVect)
    print "---"
    eigvects_per_minibatch = model.predict(d, batch_size=batch_size)[::batch_size]
    print eigvects_per_minibatch


def test_loss():
    inputs = Input(shape=(latent_dim,))
    net = Dense(latent_dim)(inputs) # linear activation
    z = BatchNormalization(name="batchnorm")(net) # enforces a per-coordinate standard normal, to avoid collapsing into zero.

    # loss_fn = dominant_eigvect_loss
    # loss_fn = random_vect_loss
    loss_fn = l2_loss
    loss = lambda x, x_pred: loss_fn(z)

    eigvec = Lambda(lambda z: dominant_eigvect_layer(z), name="eigvec")([z])

    model = Model(input=inputs, output=[z, eigvec])
    optimizer = Adam(lr=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=loss)

    N = 1000 // batch_size * batch_size
    megaepoch_count = 20

    for i in range(megaepoch_count):
        data = np.random.normal(size=(N, latent_dim))
        data[:, 0] += data[:, 1]

        print "================"
        output, eigvec = model.predict(data, batch_size=batch_size)
        print "neural eigvec pre =", eigvec[0]

        model.fit(data, [data, data], batch_size=batch_size, verbose=2)

        output, eigvec = model.predict(data, batch_size=batch_size)
        print "neural eigvec post =", eigvec[0]
        cholesky(output)


if __name__ == "__main__":
    test_loss()
    # test_eigen()
