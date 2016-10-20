'''This AE differs from a standard VAE (https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
in the following ways:

- The latent variances are not learned, they are simply set to 0.
- A normalization step takes the latent variables to a sphere surface.
- The regularization loss is changed to an energy term that
  corresponds to a pairwise repulsive force between the encoded
  elements of the minibatch.
'''

import sys
import math
import numpy as np

np.random.seed(1337)

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist


fileprefix, = sys.argv[1:]

batch_size = 100
original_dim = 784
latent_dim = 3 # one less left after normalization
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
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded)
    # Instead of a KL normality test, here's some energy function
    # that pushes the minibatch elements away from each other, pairwise.
    epsilon = 0.0001
    distances = (2.0 + epsilon - 2.0 * K.dot(z, K.transpose(z))) ** 0.5
    # regularization = -K.mean(distances) * 1000 # Keleti
    # regularization = K.mean(1.0 / distances) * 10 # Coulomb
    # regularization = 0.0 # Straight AE.
    return xent_loss + regularization


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
plt.figure(figsize=(20, 15))
points = x_test_encoded.copy()
points[:, 0] += 2.2 * (points[:, 2]>=0)
plt.scatter(points[:, 0], points[:, 1], c=y_test, s=2, lw=0)
plt.colorbar()
plt.savefig(fileprefix+"-fig1.png")

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
        # Padded with zeros at the rest of the coordinates:
        z_sample = np.array([[xi, yi, zi] + [0]*(latent_dim-3)])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig(fileprefix+"-fig2.png")

# display randomly generated digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        z_sample = np.random.normal(size=(1, latent_dim))
        z_sample /= np.linalg.norm(z_sample)
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit


plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.savefig(fileprefix+"-fig3.png")
