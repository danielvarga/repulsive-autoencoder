'''
This is experiment code for a little spinoff project of the repulsive autoencoder.

I needed an energy function with the property that in the limit, the optimal point set
has uniform density on the unit sphere. Something like the Coulomb force and the Thomson problem,
but I wanted something numerically more stable. I came up with the following:

The energy term for the interaction of two particles is their squared scalar product.
The total energy of the system is the sum over all pairs of particles.

Numerically solving the optimization problem, I've observed that with n points in m dimensions,
the optimal total energy was always exactly n^2/m. (For n>=m)

Adrian Csiszarik has managed to prove this with a simple but very attractive spectral argument.

Even without Adrian's characterization, it is easy to see in retrospect that my
proposed energy function is absolutely awful for its intended purpose of enforcing
a uniform arrangement. In contrast, if we use negative (non-squared) distance -D,
or electrostatic potential energy 1/D instead of squared scalar product,
then we do get this smoothing property. -D^2 is again in the non-uniform camp.
'''

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1337)


n = 9 # particle count
m = 3 # dimension of space
epochs = 100 # iteration count


def main():
    x = tf.Variable(np.random.normal(size=(n, m)))
    x_norm = tf.nn.l2_normalize(x, dim=1) # rows normalized
    rowwise_scalar_product = tf.matmul(x_norm, tf.transpose(x_norm))
    frobenius = tf.reduce_sum(tf.square(rowwise_scalar_product))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(frobenius)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(epochs):
            sess.run(train_step)
            print frobenius.eval(session=sess)

        x_val = x_norm.eval(session=sess)
        print "============== X"
        print x_val
        rsc = rowwise_scalar_product.eval(session=sess)

        print "============== XX^T"
        print rsc
        print "============== eigenvalues of XX^T"
        w, v = np.linalg.eig(rsc)
        print sorted(w.real.tolist(), reverse=True)
        print "============== Frobenius(XX^T)"
        print frobenius.eval(session=sess)

    interactive = True
    if interactive:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        if m==3:
            xs, ys, zs = x_val.T
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs)
        elif m==2:
            xs, ys = x_val.T
            ax = fig.add_subplot(111)
            ax.scatter(xs, ys)
        else:
            assert m in (2, 3)
        plt.show()


main()
