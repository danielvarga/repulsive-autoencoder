'''This AE differs from a standard VAE (https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
in the following ways:

- The latent variances are not learned, they are simply set to 0.
- A normalization step takes the latent variables to a sphere surface.
- The regularization loss is changed to an energy term that
  corresponds to a pairwise repulsive force between the encoded
  elements of the minibatch.
'''

import math
import numpy as np
import argparse

np.random.seed(1337)

import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import data
import vis

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest="dataset", default="mnist", help="Dataset to use: mnist/celeba")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=10, help="Number of epoch")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
args = parser.parse_args()

(x_train, x_test), (height, width) = data.load(args.dataset)


batch_size = 100
original_dim = x_test.shape[1]
intermediate_dim = 256

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)

# Completely got rid of the variational aspect
z_unnormalized = Dense(args.latent_dim)(h)

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
#    pairwise = K.sum(K.square(K.dot(z, K.transpose(z))))

    epsilon = 0.0001
    distances = (2.0 + epsilon - 2.0 * K.dot(z, K.transpose(z))) ** 0.5
    regularization = -K.mean(distances) * 1000 # Keleti
    return xent_loss + regularization


vae = Model(x, x_decoded)
vae.compile(optimizer='rmsprop', loss=vae_loss)


vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=args.nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z)

# # display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
points = x_test_encoded.copy()
points[:, 0] += 2.2 * (points[:, 2]>=0)
plt.scatter(points[:, 0], points[:, 1])
# plt.colorbar()
plt.savefig("fig1.png")

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(args.latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded = decoder(_h_decoded)
generator = Model(decoder_input, _x_decoded)

# display a 2D manifold of the digits
for y in range(1,args.latent_dim-1):
    vis.displayImageManifold(30, args.latent_dim, generator, height, width,0,y,y+1,"manifold{}".format(y))

# display randomly generated digits
vis.displayRandom(15, args.latent_dim, generator, height, width, "random")
