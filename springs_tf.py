'''
This tool lets you play with various energy functions for particle systems.
'''

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1337)


n = 100 # particle count
m = 3 # dimension of space
epochs = 500 # iteration count


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
        # print x_val
        print rowwise_scalar_product.eval(session=sess)
        print frobenius.eval(session=sess)

    interactive = True
    if interactive:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        assert m==3
        xs, ys, zs = x_val.T
        ax.scatter(xs, ys, zs)
        plt.show()


main()
