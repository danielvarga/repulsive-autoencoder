import sys
import numpy as np
import math
import keras
import keras.backend as K

# Sorry, this is tensorflow-specific, Theano and Keras
# does not even support differentiable sorting and normal.cdf
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop

from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kstest, norm

import eigen


batch_size = 256

input_dim = 2
latent_dim = 3
intermediate_dim = 200

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


def minieval(func, data):
    input = K.placeholder(shape=data.shape)
    f = K.function([input], [func(input)])
    return f([data])[0]


def test_kstest_tf():
    # Tests the hacked kstest_tf() that returns reversed_cdf.
    data = np.random.normal(size=(batch_size,))
    out = minieval(eigen.kstest_tf, data)
    top = np.sort(data)[::-1][:20]
    print top
    print norm.cdf(top, 0.0, 1.0)
    print out[:20]


def l2_loss(z):
    return L2_REG_WEIGHT * K.mean(K.square(z)) * latent_dim # sum over latent_dim, avg over minibatch


def covariance_loss(z):
    z_centered = z - K.mean(z, axis=0)
    cov = K.dot(K.transpose(z_centered), z_centered)
    loss = K.mean(K.square(K.eye(K.int_shape(z_centered)[1]) * batch_size - cov))
    return loss


# Abandoned experiment. Analogously to VAE's variance loss, d - ln d - 1 penalizes small
# positive d values harshly. Problem is, negative values can also appear.
def tuned_covariance_loss(z):
    z_centered = z - K.mean(z, axis=0)
    cov = K.dot(K.transpose(z_centered), z_centered)
    loss = K.mean(K.square(K.eye(K.int_shape(z_centered)[1]) * batch_size - cov))
    d = tf.diag(cov) / batch_size
    # diag_penalty = tf.cond(tf.greater(tf.reduce_min(d), 0), lambda: (d - K.log(d) - 1), lambda: (d-1)*(d-1))
    is_pos = tf.cast(tf.greater(d, 0), tf.float32)
    diag_penalty = is_pos * (d - K.log(d) - 1) \
        + (1 - is_pos) * 1000000
    loss += K.mean(diag_penalty)
    return loss


def matrix_exp(cov):
    cov1 = cov
    cov2 = K.dot(cov1, cov)
    cov3 = K.dot(cov2, cov)
    cov4 = K.dot(cov3, cov)
    exp_cov = K.eye(K.int_shape(cov)[1]) + cov1 + cov2/2 + cov3/6 + cov4/24
    return exp_cov


def matrix_log(cov):
    I = K.eye(K.int_shape(cov)[1])
    cov1 = cov - I
    cov2 = K.dot(cov1, cov1)
    cov3 = K.dot(cov2, cov1)
    cov4 = K.dot(cov3, cov1)
    exp_cov = I - cov1 + cov2/2 - cov3/6 + cov4/24
    return exp_cov


# Absolutely does not work at this point, probably very buggy.
def test_matrix_log():
    N = 10
    matrix = np.random.normal(size=(N, 2*N))
    matrix = np.dot(matrix, matrix.T)

    def inspect(m):
        eigVals, eigVects = np.linalg.eigh(m)
        print "eigvals =", list(reversed(eigVals))
        print "max/min eigval ratio =", eigVals[-1]/eigVals[0]
        print "dominant eigvect =", eigVects[:, -1]

    logarithm = minieval(matrix_log, matrix)
    print logarithm
    print "----"
    exponential = minieval(matrix_exp, logarithm)
    print exponential

    print "positive definite symmetric matrix"
    inspect(matrix)
    print "sum of eigenvalue logarithms", np.sum(np.log(np.linalg.eigh(matrix)[0]))
    print "====="
    print "log of matrix"
    inspect(logarithm)
    print "trace of matrix", np.trace(logarithm)


# Abandoned experiment, Taylor series of matrix_log did not converge, probably buggy as well.
def valpola_loss(z):
    z_centered = z - K.mean(z, axis=0)
    cov = K.dot(K.transpose(z_centered), z_centered)
    log_cov = matrix_log(cov)
    loss = tf.trace(cov - log_cov - K.eye(K.int_shape(z_centered)[1])) # eye is unnecessary
    return loss


def dominant_eigvect_layer(z):
    domineigvec, domineigval = eigen.eigvec_of_cov(z, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
    domineigvec_stacked = K.repeat_elements(K.reshape(domineigvec, (1, latent_dim)), batch_size, axis=0)
    # print tuple(map(int, domineigvec_stacked.get_shape()))
    return domineigvec_stacked


def cholesky(d):
    cov = np.cov(d.T)
    print "cov =", cov
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
#    optimizer = RMSprop(lr=args.lr, clipvalue=1.0)
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
    z = Dense(latent_dim, activation="linear", name="half_z")(net)
#    z = Lambda(lambda z: 2*z, name="z")(z) # increase the support to [-2,+2].
    eigvec = Lambda(lambda z: dominant_eigvect_layer(z), name="eigvec")([z])
    z_projected = Lambda(lambda z: K.reshape( K.dot(z, K.l2_normalize(K.random_normal_variable((latent_dim, 1), 0, 1), axis=-1)), (batch_size,)))([z])

    net = z
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    net = Dense(intermediate_dim, activation="relu")(net)
    net = BatchNormalization()(net)
    output = Dense(input_dim, activation="linear")(net)

    def eigenvalue_gap_loss():
        WW = K.dot(K.transpose(z), z)
        mineigval, maxeigval = eigen.extreme_eigvals(WW, batch_size, latent_dim=latent_dim, iterations=3, inner_normalization=False)
        loss = K.sqrt(maxeigval / mineigval) # K.square(maxeigval-1) + K.square(mineigval-1)
        return loss

    KSTEST_LOSS_WEIGHT = 10.0 # Keep it in sync with the aggregator!
    def kstest_loss():
        # Inner points are overrepresented in a mean, maybe reweight with some bathtub shape.
        aggregator = lambda diff: K.mean(K.square(diff))
        loss = eigen.kstest_loss(z, latent_dim, batch_size, aggregator)
        return loss

    # TODO TRY THESE, AFTER FIXING EIGENVALUE_GAP_LOSS.
    # Also, maybe something that's volume preserving, or a bit more volume-preserving.
    # EIGENVALUE_GAP_LOSS_WEIGHT = 0.1 ; shape = eigenvalue_gap_loss(x, x_pred) * EIGENVALUE_GAP_LOSS_WEIGHT
    # RANDOM_VECT_LOSS_WEIGHT = 10 ; shape = random_vect_loss(z) * RANDOM_VECT_LOSS_WEIGHT

    loss_name = "covariance"

    if loss_name == "kstest":
        loss_fn = kstest_loss
        LOSS_WEIGHT = KSTEST_LOSS_WEIGHT
    elif loss_name == "eigenvalue_gap":
        loss_fn = eigenvalue_gap_loss
        LOSS_WEIGHT = EIGENVALUE_GAP_LOSS_WEIGHT
    elif loss_name == "l2":
        loss_fn = lambda: K.mean(K.square(z))
        LOSS_WEIGHT = 1.0
    elif loss_name == "0":
        loss_fn = lambda: K.mean(K.square(z))*0.0
        LOSS_WEIGHT = 0.0
    elif loss_name == "covariance":
        loss_fn = lambda: covariance_loss(z)
        LOSS_WEIGHT = 0.02 / 256
    elif loss_name == "tuned_covariance":
        loss_fn = lambda: tuned_covariance_loss(z)
        LOSS_WEIGHT = 0.02
    elif loss_name == "valpola":
        loss_fn = lambda: valpola_loss(z)
        LOSS_WEIGHT = 1.0
    else:
        assert False, "unknown loss name"

    print "loss:", loss_name, "weight:", LOSS_WEIGHT

    def total_loss(x, x_pred):
        recons_loss = K.mean(K.square(x-x_pred))
        return recons_loss + LOSS_WEIGHT * loss_fn()

    model = Model(input=inputs, output=output)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=total_loss,
                  metrics=["mse", lambda _1, _2: loss_fn() ])

    encoder = Model(input=inputs, output=z)
    encoder.compile(optimizer=optimizer, loss="mse")

    N = 10000 // batch_size * batch_size
    epoch_count = 10
    megaepoch_count = 5

    def sampler_uniform(n, d):
        return np.random.uniform(size=(n, d)) * 2 - 1

    # 2 cubes
    def sampler_bicube(n, d):
        return np.random.uniform(size=(n, d)) * \
            np.expand_dims((np.random.randint(2, size=n).astype(np.float32) * 2 - 1), 1)

    # 2^d cubes [1/3, 1]^d
    def sampler_expcube(n, d):
        return ((np.random.uniform(size=(n, d)) * 2 + 1) / 3) * (np.random.randint(2, size=(n,d)).astype(np.float32) * 2 - 1)

    datasets = {"uniform": sampler_uniform, "bicube": sampler_bicube, "expcube": sampler_expcube}

    dataset = "uniform"
    print "dataset:", dataset
    sampler = datasets[dataset]

    for i in range(megaepoch_count):
        print "================"
        data = sampler(N, input_dim)
        model.fit(data, data, epochs=epoch_count, batch_size=batch_size, verbose=2)
        data = sampler(N, input_dim)
        z = encoder.predict(data, batch_size=batch_size)
        output = model.predict(data, batch_size=batch_size)
        cholesky(z)

    for i in range(min((latent_dim, 10))):
        print "KS for dim", i, "=", kstest(z[:, i], 'norm')
    print "histogram for standard normal"
    print np.histogram(np.random.normal(size=N), 20)
    for i in range(min((latent_dim, 10))):
        print "---"
        projector = np.random.normal(size=(latent_dim,))
        projector /= np.linalg.norm(projector)
        projected_z = z.dot(projector)
        print projected_z.shape
        projected_z = projected_z.flatten()
        print np.histogram(projected_z, 20)
        print "KS for random projection", i, "=", kstest(projected_z, 'norm')

    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D

    plt.scatter(output[:, 0], output[:, 1])
    plt.show()

    if latent_dim==1:
        plt.hist(z, bins=20)
    elif latent_dim==2:
        plt.scatter(z[:, 0], z[:, 1])
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z = z[:2000, :]
        # TODO color according to reconstruction loss.
        ax.scatter(z[:, 0], z[:, 1], z[:, 2])
        ax.set_title('Latent points')
    plt.show()

    def cumulative_view(projected_z, title):
        fig, ax = plt.subplots(figsize=(10, 8))
        n, bins, patches = ax.hist(projected_z, bins=100, cumulative=True,
                                    normed=1, histtype='step', label='Empirical')
        mu = np.mean(projected_z)
        sigma = np.std(projected_z)
        y = norm.cdf(bins, mu, sigma)
        ax.plot(bins, y, 'k--', linewidth=1.5, label='Fitted normal')
        y = norm.cdf(bins, 0.0, 1.0)
        ax.plot(bins, y, 'r--', linewidth=1.5, label='Standard normal')
        ax.grid(True)
        ax.legend(loc='lower right')
        ax.set_title(title)
        plt.show()

    for i in range(1, 4):
        projector = np.random.normal(size=(latent_dim,))
        projector /= np.linalg.norm(projector)
        projected_z = z.dot(projector)
        cumulative_view(projected_z, "CDF of randomly projected latent cloud #%d" % i)
    cumulative_view(z[:, 0], "CDF of first coordinate of latent cloud")
    cumulative_view(z[:, 1], "CDF of second coordinate of latent cloud")


def test_kstest():
    pass


if __name__ == "__main__":
    # test_eigen()
    test_loss()
    # test_kstest()
