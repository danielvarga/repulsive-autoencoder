import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU

import model_dcgan
import net_blocks
import callbacks

np.random.seed(100)

epsilon = 0.0001
trainSize = 50000
disc_size = "small"
wd = 0.0
use_bn = True
batch_size = 200
dense_dims = [100,100, 100, 100]
activation = "relu"
clipValue = 0.01
gradient_weight = 100.0
verbose=1
prefix = "pictures/testdisc"
dim=1

if dim == 1:
    # x_train = 10.0 * np.sign(np.random.randint(0,3, size=trainSize) - 1.5) # -10 with 2/3 probability and 10 with 1/3 probability
    x_values = np.array([-10, 0, 0, 10, 10])
    x_train = x_values[np.random.randint(0,5, size=trainSize)]
    x_generated = np.random.uniform(-10, 10, size=trainSize) # uniform between -10 and 10
    test_points = np.linspace(-20, 20, batch_size)
if dim == 2:
    # x_train = 10.0 * np.sign(np.random.randint(0,3, size=trainSize) - 1.5) # -10 with 2/3 probability and 10 with 1/3 probability
    x_values = np.array([[-10,-10], [-10,10], [0,0], [0,0], [10,-10], [10, 10]])
    x_train = x_values[np.random.randint(0,6, size=trainSize)]
    x_generated = np.random.uniform(-10, 10, size=(trainSize, 2))
    x_coords = np.linspace(-10, 10, batch_size)
    test_points = np.transpose([np.tile(x_coords, len(x_coords)), np.repeat(x_coords, len(x_coords))])

# x_generated = 3.0 * np.random.normal(size=trainSize) + x_train

y_train = np.ones(shape=[len(x_train)])
y_generated = - y_train
xs = np.concatenate([x_train, x_generated])
ys = np.concatenate([y_train, y_generated])

#discriminator_channels = model_dcgan.default_channels("discriminator", disc_size, None)
#disc_layers = model_dcgan.discriminator_layers_wgan(discriminator_channels, wd=wd, bn_allowed=use_bn)
disc_layers = net_blocks.dense_block(dense_dims, wd, use_bn, activation)
disc_layers.append(Dense(1, activation="linear"))


disc_input = Input(batch_shape=[batch_size, dim], name="disc_input")
disc_output = disc_input
for layer in disc_layers:
    disc_output = layer(disc_output)

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    loss = - K.mean(y_true * y_pred)
    return loss
def grad_aux(output, input):
    grads = K.gradients(output, [input])
    grads = grads[0]
    tensor_axes = range(1, K.ndim(grads))
    grads = K.sqrt(epsilon + K.sum(K.square(grads), axis=tensor_axes))
    return grads
def grad_flat(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.square(K.maximum(k1, grads) - k1))
    return grad_penalty
def grad_orig(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.square(grads - k1))
    return grad_penalty
def grad_hill(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.abs(K.square(grads) - k1))
    return grad_penalty


def hill(y_true, y_pred):
    return D_loss(y_true, y_pred) + gradient_weight * grad_hill(y_true, y_pred)
def orig(y_true, y_pred):
    return D_loss(y_true, y_pred) + gradient_weight * grad_orig(y_true, y_pred)
def flat(y_true, y_pred):
    return D_loss(y_true, y_pred) + gradient_weight * grad_flat(y_true, y_pred)

metrics = [D_loss, grad_flat, grad_orig, grad_hill]

discriminator = Model(disc_input, disc_output)
optimizer = RMSprop()
discriminator.compile(optimizer=optimizer, loss="mse", metrics=metrics)
discriminator.summary()
discriminator.save_weights("a.h5")

losses = [orig, hill, flat]
epochs = [5, 5, 5]
count = len(losses)
predictions = []

for i in range(count):
    loss = losses[i]
    nb_epoch = epochs[i]
    optimizer = RMSprop()
    discriminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    discriminator.load_weights("a.h5")
    discriminator.fit(xs, ys,
                      verbose=verbose,
                      shuffle=True,
                      epochs = nb_epoch,
                      batch_size=batch_size
    )
    predictions.append(discriminator.predict(test_points, batch_size=batch_size))

from pylab import *
colmap = cm.ScalarMappable(cmap=cm.hsv)

# display result
name = prefix + "-output.png"
print "Saving {}".format(name)
fig = plt.figure()
for i in range(count):
    if dim == 1:
        ax = fig.add_subplot(count, 1, i+1)
 #       f, axes = plt.subplots(count, 1, sharex=True)
        ax.scatter(test_points, predictions[i])
    elif dim == 2:
        ax = fig.add_subplot(count, 1, i+1, projection='3d')
        #        ax.scatter(test_points[:,0], test_points[:,1], predictions[i])
        ax.scatter(test_points[:,0], test_points[:,1], predictions[i], marker='o')
plt.savefig(name)
plt.close()
