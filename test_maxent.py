import sys
import numpy as np
import keras.backend as K
import tensorflow as tf


# all length l tuples formed from the set [d]
def power(d, l):
    if l==1:
        return [ (m,) for m in range(d) ]

    low = power(d, l-1)
    l = []
    for i in range(d):
        l += [tuple([i]+list(tu)) for tu in low]
    return l


# probably obsoleted by weighted version
def momenta_of_cloud(data, l):
    n, d = data.shape
    momenta = np.zeros(tuple([d]*l))
    for tup in power(d, l): # len(tup)=l
        muls = np.prod(data[:, tup], axis=1)
        assert muls.shape==(n,)
        moment = muls.mean()
        momenta[tup] = moment
    return momenta


# probably obsoleted by weighted version
# rows are data points, <=k momenta.
def moment_signature_of_cloud(data, k):
    signature = []
    for l in range(1, k+1):
        signature.append(momenta_of_cloud(data, l))
    return signature


def momenta_of_weighted_cloud(data, weights, l):
    n, d = data.shape
    assert weights.shape == (n,)
    momenta = np.zeros(tuple([d]*l))
    for tup in power(d, l): # len(tup)=l
        muls = np.prod(data[:, tup], axis=1)
        assert muls.shape==(n,)
        moment = (muls * weights).sum()
        momenta[tup] = moment
    return momenta


def momenta_of_weighted_cloud_tf_1(data, weights):
    w = data * K.expand_dims(weights, 1)
    return K.sum(w, axis=0)


def momenta_of_weighted_cloud_tf_2(data, weights):
    w = data * K.expand_dims(weights, 1)
    return K.dot(K.transpose(data), w)


def moment_signature_of_weighted_cloud_tf_2(data, weights):
    return [momenta_of_weighted_cloud_tf_1(data, weights), momenta_of_weighted_cloud_tf_2(data, weights)]


def test_moment_signature_tf_2():
    d = 2
    m = 100
    data = grid(d, m)
    # data = np.array([[1, 1], [1, 2]]).astype(float)
    n = data.shape[0]
    weights = np.random.uniform(size=n)
    weights /= weights.sum()

    data_var = K.placeholder(shape=data.shape)
    weights_var = K.placeholder(shape=weights.shape)

    signia_vars = moment_signature_of_weighted_cloud_tf_2(data_var, weights_var)
    signia_fn = K.function(inputs=[data_var, weights_var], outputs=signia_vars)
    print "numpy-calculated signature:"
    print moment_signature_of_weighted_cloud(data, weights, 2)
    print "GPU calculated signature:"
    print signia_fn([data, weights])


def moment_signature_of_weighted_cloud(data, weights, k):
    signature = []
    for l in range(1, k+1):
        signature.append(momenta_of_weighted_cloud(data, weights, l))
    return signature


# for simplicity, the support is [0,1]^d, consisting of cubes of side 1/m.
# Should translate and scale input so that this support is large enough.
# 
# mgrid[[slice]] is a bit cryptic, see:
# https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy/32208788#32208788
# https://stackoverflow.com/questions/28825219/how-can-i-create-an-n-dimensional-grid-in-numpy-to-evaluate-a-function-for-arbit/28825910#28825910
def grid(d, m):
    g = np.mgrid[[slice(m)] * d]
    g = g.reshape(d, -1).T
    return g.astype(np.float32) / m


def entropy(weights):
    return np.mean(weights * np.log(weights))


def entropy_tf(weights):
    return K.mean(weights * K.log(weights))


def test_signature():
    data = np.array([[0,1], [1,0], [1,1]]).astype(np.float32)
    n = data.shape[0]
    weights = np.ones((n, )) / n
    k = 3
    signia = moment_signature_of_weighted_cloud(data, weights, k)
    for momenta in signia:
        print momenta
        print


def test_signature_of_grid():
    d = 2
    m = 100
    g = grid(d, m)
    n = g.shape[0]
    weights = np.random.uniform(size=n)
    weights /= weights.sum()
    k = 3
    signia = moment_signature_of_weighted_cloud(g, weights, k)
    for momenta in signia:
        print momenta
        print


def optimize_grid():
    d = 2
    target_data = np.array([[1./3,2./3], [2./3,1./3], [2./3,2./3]]).astype(np.float32)
    target_n = target_data.shape[0]
    target_weights = np.ones((target_n, )) / target_n
    k = 2
    target_signia = moment_signature_of_weighted_cloud(target_data, target_weights, k)

    m = 10
    data = grid(d, m).astype(np.float32)
    n = data.shape[0]
    weights = np.random.uniform(size=n).astype(np.float32)
    weights /= weights.sum()

    data_var = tf.Variable(initial_value=data, trainable=False)
    weights_var = tf.Variable(initial_value=weights, trainable=True)

    signia_vars = moment_signature_of_weighted_cloud_tf_2(data_var, weights_var)
    signia_fn = K.function(inputs=[data_var, weights_var], outputs=signia_vars)

    signia_loss = K.mean((signia_vars[0] - target_signia[0]) ** 2) + K.mean((signia_vars[0] - target_signia[0]) ** 2)
    entropy_loss = entropy_tf(weights_var)
    ENTROPY_LOSS_WEIGHT = 1.0
    total_loss = signia_loss + ENTROPY_LOSS_WEIGHT * (-entropy_loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    train_step = optimizer.minimize(total_loss)

    epochs = 2000

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(epochs):
            sess.run(train_step)
            if step % 100 == 0:
                weights_val = weights_var.eval(session=sess)
                print step, signia_loss.eval(session=sess), entropy_loss.eval(session=sess), weights_val.sum()
                if weights_val.min() <= 0.0:
                    print "underflow"
                    weights_val = np.clip(weights_val, 1e-6, 1)
                weights_val /= weights_val.sum()
                weights_var.load(weights_val, sess)
        vis = weights_val.reshape((m, m))
        import matplotlib.pyplot as plt
        plt.imshow(vis)
        plt.show()
        plt.savefig("vis.png")


def main():
    # test_moment_signature_tf_2()
    # test_giniture_of_grid()
    optimize_grid()


main()
