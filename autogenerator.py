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
import callbacks

import model
import model_conv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest="dataset", default="mnist", help="Dataset to use: mnist/celeba")
parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=10, help="Number of epochs")
parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")
parser.add_argument('--intermediate_dims', dest="intermediate_dims_string", default="256", help="Intermediate dimensions")
parser.add_argument('--frequency', dest="frequency", type=int, default=10, help="image saving frequency")
parser.add_argument('--model', dest="model", default="rae", help="Model to use: rae/vae/nvae/vae_conv")
parser.add_argument('--output', dest="prefix", help="File prefix for the output visualizations and models.")

args = parser.parse_args()

assert args.prefix is not None, "Please specify an output file prefix with the --output arg."

assert args.model in ("rae", "vae", "nvae", "vae_conv", "nvae_conv", "universal"), "Unknown model type."
print("Training model of type %s" % args.model)

(x_train, x_test), (height, width) = data.load(args.dataset)


batch_size = 500
original_dim = x_test.shape[1]
intermediate_dims = list(map(int, args.intermediate_dims_string.split(",")))

# Using modules where normal people would use classes.
if args.model == "rae":
    sampler = model.spherical_sampler
    dense_encoder = model.DenseEncoder(intermediate_dims)
    dense_decoder = model.DenseDecoder(args.latent_dim, intermediate_dims, original_dim)
    nonvariational = True
    vae, encoder, generator = model.build_model(
                                    batch_size, original_dim,
                                    dense_encoder, args.latent_dim, dense_decoder,
                                    nonvariational=nonvariational,
                                    spherical=True)
elif args.model in ("vae", "nvae"):
    sampler = model.gaussian_sampler
    dense_encoder = model.DenseEncoder(intermediate_dims)
    dense_decoder = model.DenseDecoder(args.latent_dim, intermediate_dims, original_dim)
    nonvariational = args.model=="nvae"
    vae, encoder, generator = model.build_model(
                                    batch_size, original_dim,
                                    dense_encoder, args.latent_dim, dense_decoder,
                                    nonvariational=nonvariational)
elif args.model in ("vae_conv", "nvae_conv"):
    sampler = model.gaussian_sampler
    # def __init__(self, levels_config, filter_num_config, latent_dim, img_size, activation_config=None, batch_size=32, wd=0.003):

    conv_encoder = model_conv.ConvEncoder([2,2,2], [32,32,32], 20, [60, 72, 1], batch_size=batch_size)
    conv_decoder = model_conv.ConvDecoder([2,2,2], [32,32,32], 20, [60, 72, 1], batch_size=batch_size)

    nonvariational = args.model=="nvae_conv"
    vae, encoder, generator = model.build_model(
                                    batch_size, original_dim,
                                    conv_encoder, args.latent_dim, conv_decoder,
                                    nonvariational=nonvariational)
   
else:
    assert False, "model type %s not yet implemented, please be patient." % args.model

vae.summary()






gen_original_dim=args.latent_dim
gen_sampler = model.gaussian_sampler
gen_dense_encoder = model.DenseEncoder([20])
gen_latent_dim = 20

gen_dense_decoder = model.DenseDecoder(gen_latent_dim, intermediate_dims, gen_original_dim)
gen_nonvariational = args.model=="nvae"
gen_nonvariational = False
gen_vae, gen_encoder, gen_generator = model.build_model(
                                batch_size, gen_original_dim,
                                gen_dense_encoder, gen_latent_dim, gen_dense_decoder,
                                nonvariational=gen_nonvariational)



cbs = []
cbs.append(callbacks.get_lr_scheduler(args.nb_epoch))
cbs.append(callbacks.imageDisplayCallback(
    x_train, x_test,
    args.latent_dim, batch_size, height, width,
    encoder, generator, sampler,
    args.prefix, args.frequency))

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=args.nb_epoch,
        batch_size=batch_size,
        callbacks = cbs,
        validation_data=(x_test, x_test))

vae.save("model_%s.h5" % args.prefix)
encoder.save("model_%s.h5" % args.prefix)
generator.save("model_%s.h5" % args.prefix)

zs = encoder.predict(x_train, batch_size=batch_size)
zs_test = encoder.predict(x_test, batch_size=batch_size)


gen_cbs = []
gen_cbs.append(callbacks.get_lr_scheduler(args.nb_epoch))
gen_vae.fit(zs, zs,
        shuffle=True,
        nb_epoch=args.nb_epoch,
        batch_size=batch_size,
        callbacks = gen_cbs,
        validation_data=(zs_test, zs_test))


z_sample = sampler(batch_size, gen_latent_dim)
x_decoded = gen_generator.predict(z_sample, batch_size=batch_size)

def gen_sampler(latent_dim, batch_size):
    return x_decoded

vis.displayRandom(15, args.latent_dim, gen_sampler, generator, height, width, "%s-rando-gen" % args.prefix, batch_size=batch_size)





#vae.save("model_%s.h5" % args.prefix)

# # display a 2D plot of the validation set in the latent space
# vis.latentScatter(encoder, x_test, batch_size, args.prefix+"-fig1")

# display 2D manifolds of images
show_manifolds = False
if show_manifolds:
    for y in range(1, args.latent_dim-1):
        vis.displayImageManifold(30, args.latent_dim, generator, height, width, 0, y, y+1, "%s-manifold%d" % (args.prefix, y), batch_size=batch_size)

# display randomly generated images
vis.displayRandom(15, args.latent_dim, sampler, generator, height, width, "%s-random" % args.prefix, batch_size=batch_size)

vis.displaySet(x_test[:batch_size], 100, vae, height, width, "%s-test" % args.prefix)
vis.displaySet(x_train[:batch_size], 100, vae, height, width, "%s-train" % args.prefix)

# display image interpolation
vis.displayInterp(x_train, x_test, batch_size, args.latent_dim, height, width, encoder, generator, 10, "%s-interp" % args.prefix)

