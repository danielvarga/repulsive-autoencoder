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
import model_rae

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest="dataset", default="mnist", help="Dataset to use: mnist/celeba")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=10, help="Number of epoch")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
args = parser.parse_args()

(x_train, x_test), (height, width) = data.load(args.dataset)


batch_size = 100
original_dim = x_test.shape[1]
intermediate_dim = 256

vae, encoder, generator = model_rae.build_rae(batch_size, original_dim, intermediate_dim, args.latent_dim)

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=args.nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# # display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
points = x_test_encoded.copy()
points[:, 0] += 2.2 * (points[:, 2]>=0)
plt.scatter(points[:, 0], points[:, 1])
# plt.colorbar()
plt.savefig("fig1.png")

# display a 2D manifold of the digits
for y in range(1,args.latent_dim-1):
    vis.displayImageManifold(30, args.latent_dim, generator, height, width,0,y,y+1,"manifold{}".format(y))

# display randomly generated digits
vis.displayRandom(15, args.latent_dim, generator, height, width, "random")
