'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''

import math
import numpy as np

np.random.seed(1337)

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


batch_size = 500
original_dim = 784
latent_dim = 3 # 2 are left after normalization
intermediate_dim = 256
nb_epoch = 50

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)

# Completely got rid of the variational aspect
z_unnormalized = Dense(latent_dim)(h)

# normalize all latent vars:
z = Lambda(lambda z_unnormed: K.l2_normalize(z_unnormed, axis=-1))([z_unnormalized])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded = decoder(h_decoded)


def vae_loss(x, x_decoded):
    # DANIEL Why not mean_squared_error?
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
    # Instead of a KL normality test, let's try some energy function
    # that pushes the minibatch element away from each other, pairwise.
    pairwise = K.sum(K.square(K.dot(z, K.transpose(z))))
    return xent_loss + pairwise / 1000 # Maybe some broadcasting happens here?


vae = Model(x, x_decoded)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
points = x_test_encoded.copy()
points[:, 0] += 2.2 * (points[:, 2]>=0)
plt.scatter(points[:, 0], points[:, 1], c=y_test)
plt.colorbar()
plt.savefig("fig1.png")

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded = decoder(_h_decoded)
generator = Model(decoder_input, _x_decoded)

# display a 2D manifold of the digits
n = 30  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = np.linspace(-1, +1, n)
grid_y = np.linspace(-1, +1, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        zisqr = 1.000001-xi*xi-yi*yi
        if zisqr < 0.0:
            continue
        zi = math.sqrt(zisqr)
        z_sample = np.array([[xi, yi, zi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig("fig2.png")
