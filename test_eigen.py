import numpy as np
import math
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kstest

import eigen


batch_size = 256

input_dim = 3
latent_dim = 3
intermediate_dim = 100

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
    domineigvec, domineigval = eigen.eigvec_of_cov(z, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
    loss = K.square(K.dot(z, domineigvec))
    loss = K.mean(loss)
    return loss


# input: 1d tensor. output: ks test statistic.
def kstest_tf(p):
    import tensorflow
    values, indices = tf.nn.top_k(p, k=0, sorted=True)
    return K.mean(K.square(values))


def kstest_loss(z):
    v = K.random_normal_variable((latent_dim, 1), 0, 1)
    v = v / K.sqrt(K.dot(K.transpose(v), v))
    p = K.dot(z, v)
    return kstest_tf(p)


def l2_loss(z):
    return L2_REG_WEIGHT * K.mean(K.square(z)) * latent_dim # sum over latent_dim, avg over minibatch


def dominant_eigvect_layer(z):
    domineigvec, domineigval = eigen.eigvec_of_cov(z, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
    domineigvec_stacked = K.repeat_elements(K.reshape(domineigvec, (1, latent_dim)), batch_size, axis=0)
    # print tuple(map(int, domineigvec_stacked.get_shape()))
    return domineigvec_stacked


def cholesky(d):
    cov = np.cov(d.T)
    mean = np.mean(d, axis=0)

    print "means =", mean
    print "deviations = ", np.std(d, axis=0)
    print "data sample = "
    print d[:5]

    eigVals, eigVects = np.linalg.eigh(cov)
    print "sqrt eigvals =", list(reversed(np.sqrt(eigVals)))
    print "sqrt of max/min eigval ratio =", math.sqrt(eigVals[-1]/eigVals[0])
    print "dominant eigvect =", eigVects[:, -1]

    return

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


# Autoencoder that maps uniform -> gaussian -> uniform.
def test_loss():
    inputs = Input(shape=(input_dim,))
    net = inputs
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    z = Dense(latent_dim, activation="tanh", name="z")(net)
    eigvec = Lambda(lambda z: dominant_eigvect_layer(z), name="eigvec")([z])
    net = z
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    output = Dense(latent_dim, activation="sigmoid")(net)

    def eigenvalue_gap_loss(x, x_pred):
        WW = K.dot(K.transpose(z), z)
        mineigval, maxeigval = eigen.extreme_eigvals(WW, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
        loss = K.sqrt(maxeigval / mineigval) # K.square(maxeigval-1) + K.square(mineigval-1)
        return loss

    def total_loss(x, x_pred):
        recons = K.mean(K.square(x-x_pred))
        shape = K.mean(K.square(z)) / 10
        # EIGENVALUE_GAP_LOSS_WEIGHT = 0.1 ; shape += eigenvalue_gap_loss(x, x_pred) * EIGENVALUE_GAP_LOSS_WEIGHT
        # RANDOM_VECT_LOSS_WEIGHT = 10 ; shape += random_vect_loss(z) * RANDOM_VECT_LOSS_WEIGHT
        return recons + shape

    model = Model(input=inputs, output=output)
    optimizer = Adam(lr=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss=total_loss,
        metrics=["mse", eigenvalue_gap_loss, lambda _1, _2: random_vect_loss(z)])

    encoder = Model(input=inputs, output=z)
    encoder.compile(optimizer=optimizer, loss="mse")

    N = 1000 // batch_size * batch_size
    megaepoch_count = 1

    for i in range(megaepoch_count):
        print "================"
        data = np.random.uniform(size=(N, input_dim)) * 2 - 1
        model.fit(data, data, batch_size=batch_size, verbose=2)
        data = np.random.uniform(size=(N, input_dim)) * 2 - 1
        z = encoder.predict(data, batch_size=batch_size)
        # cholesky(z)

    projector = GaussianRandomProjection(n_components=1)

    for i in range(latent_dim):
        print "KS for dim", i, "=", kstest(z[:, i], 'norm')
    for i in range(latent_dim):
        print "KS for random projection", i, "=", kstest(projector.fit_transform(z), 'norm')

    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D

    if latent_dim==1:
        plt.hist(z, bins=20)
    elif latent_dim==2:
        plt.scatter(z[:, 0], z[:, 1])
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z = z[:1000, :]
        # TODO color according to reconstruction loss.
        ax.scatter(z[:, 0], z[:, 1], z[:, 2])
    plt.show()

    def cumulative_view(projected_z, title):
        fig, ax = plt.subplots(figsize=(8, 4))
        n, bins, patches = ax.hist(projected_z, bins=20, cumulative=True,
                                    normed=1, histtype='step', label='Empirical')
        from scipy.stats import norm
        mu = np.mean(projected_z)
        sigma = np.std(projected_z)
        y = norm.cdf(bins, mu, sigma)
        ax.plot(bins, y, 'k--', linewidth=1.5, label='Fitted normal')
        y = norm.cdf(bins, 0.0, 1.0)
        ax.plot(bins, y, 'r--', linewidth=1.5, label='Standard normal')
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title(title)
        plt.show()

    cumulative_view(projector.fit_transform(z), "CDF of randomly projected latent cloud")
    cumulative_view(z[:, 0], "CDF of first coordinate of latent cloud")


def test_kstest():
    pass


if __name__ == "__main__":
    # test_eigen()
    test_loss()
    # test_kstest()
