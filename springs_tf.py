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
epochs = 1000 # iteration count


def squaredDistanceTest():
    l_a = 8
    l_b = 9
    dim = 10
    a = tf.Variable(np.random.normal(size=(l_a, dim)))
    b = tf.Variable(np.random.normal(size=(l_b, dim)))
    aL2S = tf.reduce_sum(a**2, -1, keep_dims=True)
    bL2S = tf.reduce_sum(b**2, -1, keep_dims=True)
    aL2SM = tf.reshape(tf.tile(aL2S, (l_b, 1)), (l_b, l_a))
    bL2SM = tf.reshape(tf.tile(bL2S, (l_a, 1)), (l_a, l_b))
    print(aL2SM.get_shape().as_list())
    print(bL2SM.get_shape().as_list())
    squaredDistances = aL2SM + tf.transpose(bL2SM) - 2.0 * tf.matmul(b, tf.transpose(a))
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        a_val = a.eval(session=sess)
        b_val = b.eval(session=sess)
        sd = squaredDistances.eval(session=sess)
        print(sd)
        print("----")
        print(np.array([[(al-bl).dot(al-bl) for al in a_val] for bl in b_val]))


def main():
    x = tf.Variable(np.random.normal(size=(n, m)))
    x_norm = tf.nn.l2_normalize(x, dim=1) # rows normalized

    epsilon = 0.0001
    distances = (2.0 + epsilon - 2.0 * tf.matmul(x_norm, tf.transpose(x_norm))) ** 0.5

    energy = -tf.reduce_sum(distances)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(energy)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in range(epochs):
            sess.run(train_step)
            print(energy.eval(session=sess))

        x_val = x_norm.eval(session=sess)
        print("============== X")
        print(x_val)
        distances_val = distances.eval(session=sess)

        print("============== Pairwise distances")
        print(distances_val)

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
