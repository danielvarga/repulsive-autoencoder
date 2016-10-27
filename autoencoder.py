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

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import data
import vis
import model_rae
import model_vae

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest="dataset", default="mnist", help="Dataset to use: mnist/celeba")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=10, help="Number of epochs")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
parser.add_argument('--intermediate_dims', dest="intermediate_dims_string", default="256", help="Intermediate dimensions")
parser.add_argument('--model', dest="model", default="rae", help="Model to use: rae/vae/nvae/vae_conv")
parser.add_argument('--output', dest="prefix", help="File prefix for the output visualizations and models.")

args = parser.parse_args()

assert args.prefix is not None, "Please specify an output file prefix with the --output arg."

assert args.model in ("rae", "vae", "nvae", "vae_conv"), "Unknown model type."
print "Training model of type %s" % args.model

(x_train, x_test), (height, width) = data.load(args.dataset)


batch_size = 1000
original_dim = x_test.shape[1]
intermediate_dims = map(int, args.intermediate_dims_string.split(","))

# Using modules where normal people would use classes.
if args.model == "rae":
    model_module = model_rae
    vae, encoder, generator = model_module.build_model(batch_size, original_dim, intermediate_dims, args.latent_dim)
elif args.model == "vae":
    model_module = model_vae
    vae, encoder, generator = model_module.build_model(batch_size, original_dim, intermediate_dims, args.latent_dim, nonvariational=False)
elif args.model == "nvae":
    model_module = model_vae
    vae, encoder, generator = model_module.build_model(batch_size, original_dim, intermediate_dims, args.latent_dim, nonvariational=True)
else:
    assert False, "model type %s not yet implemented, please be patient." % args.model

vae.summary()
vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=args.nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


# # display a 2D plot of the validation set in the latent space
# vis.latentScatter(encoder, x_test, batch_size, args.prefix+"-fig1")

# display 2D manifolds of images
show_manifolds = False
if show_manifolds:
    for y in range(1, args.latent_dim-1):
        vis.displayImageManifold(30, args.latent_dim, generator, height, width, 0, y, y+1, "%s-manifold%d" % (args.prefix, y))

# display randomly generated images
vis.displayRandom(15, args.latent_dim, model_module.sample, generator, height, width, "%s-random" % args.prefix)

vis.displaySet(x_test[:batch_size], 100, vae, height, width, "%s-test" % args.prefix)
vis.displaySet(x_train[:batch_size], 100, vae, height, width, "%s-train" % args.prefix)

####
vis.displayInterp(x_train, x_test, batch_size, args.latent_dim, height, width, encoder, generator, 10, "%s-interp" % args.prefix)
