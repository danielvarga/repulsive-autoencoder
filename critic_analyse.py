import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import data
import vis
import load_models
from keras import backend as K


import dcgan_params
args = dcgan_params.getArgs()
print(args)

# set random seed
np.random.seed(10)

sample_size = 1000

prefix = args.prefix
discriminator_prefix = args.prefix + "_discriminator"
generator_prefix = args.prefix + "_generator"
gendisc_prefix = args.prefix + "_gendisc"

discriminator = load_models.loadModel(discriminator_prefix)
generator = load_models.loadModel(generator_prefix)
gendisc = load_models.loadModel(gendisc_prefix)

# interpolate between latent points
latent_samples = np.random.normal(size=(2, args.latent_dim))
vis.interpBetween(latent_samples[0], latent_samples[1], generator, args.batch_size, prefix + "_interpBetween")

data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(sample_size, sample_size)
noise_samples = np.random.uniform(size=x_train.shape)
generated_samples = generator.predict(np.random.normal(size=(x_train.shape[0], args.latent_dim)), batch_size=args.batch_size)
positive = discriminator.predict(x_train, batch_size=args.batch_size)
noise = discriminator.predict(noise_samples, batch_size=args.batch_size)
generated = discriminator.predict(generated_samples, batch_size=args.batch_size)


# compare the discriminator with respect to the real images, generated images, random noise
plt.figure(figsize=(12,12))
plt.hist(positive, label='images', alpha=0.5, bins=100)
plt.hist(noise, label='noise', alpha=0.5, bins=100)
plt.hist(generated, label='generated', alpha=0.5, bins=100)
plt.legend()
fileName = prefix + "_posneg.png"
print("Creating file: " + fileName)
plt.savefig(fileName)
plt.close()

# sort train and generated images according to their discriminator ranking
positive_sorter = np.argsort(np.concatenate([positive[:,0], generated[:,0]]))
x_train_sorted = np.concatenate([x_train, generated_samples])[positive_sorter]
vis.plotImages(x_train_sorted[::5], 20, 20, prefix + "_discriminator_order")

# check what happens if we transform the images
x_train_upside_down = x_train[:,::-1]
x_train_inverted = 1-x_train
upside_down = discriminator.predict(x_train_upside_down, batch_size=args.batch_size)
inverted = discriminator.predict(x_train_inverted, batch_size=args.batch_size)
plt.figure(figsize=(12,12))
plt.hist(positive, label='images', alpha=0.5, bins=100)
plt.hist(upside_down, label='upside_down', alpha=0.5, bins=100)
plt.hist(inverted, label='inverted', alpha=0.5, bins=100)
plt.legend()
fileName = prefix + "_transformed.png"
print("Creating file: " + fileName)
plt.savefig(fileName)
plt.close()

# visualize weight magnitudes for conv and bn layers
plt.figure(figsize=(12,12))
conv_layers = [1, 3, 6, 9, 12]
for i in conv_layers:
    weights = discriminator.layers[i].get_weights()
    w = np.array(weights)
    w = w.reshape((np.prod(w.shape),))
    plt.hist(w, label = "conv_layer_{}".format(i), alpha=0.5, bins=100)
plt.legend()
fileName = prefix + "_conv_weight_hist.png"
print("Creating file: " + fileName)
plt.savefig(fileName)    
plt.close()


plt.figure(figsize=(12,12))
bn_layers = [4, 7, 10]
f, axarr = plt.subplots(len(bn_layers), 2)
axarr[0,0].set_title("bn_layers_gamma")
axarr[0,1].set_title("bn_layers_beta")
for i, l in enumerate(bn_layers):
    weights = discriminator.layers[l].get_weights()
    axarr[i,0].hist(weights[0], alpha=0.5, bins=100)
    axarr[i, 0].locator_params(nbins=3, axis='x')
#    plt.xlim(-0.01, 0.01)

    axarr[i,1].hist(weights[1], alpha=0.5, bins=100)
    axarr[i, 1].locator_params(nbins=3, axis='x')
#    plt.xlim(-0.01, 0.01)
#    w = np.array(weights[:2])
#    w = w.reshape((np.prod(w.shape),))
# plt.legend()
fileName = prefix + "_bn_weight_hist.png"
print("Creating file: " + fileName)
plt.savefig(fileName)    
plt.close()

# visualize activation magnitudes
plt.figure(figsize=(12,12))
#conv_layers = [1, 3, 6, 9, 12]
i = 0
for layer in discriminator.layers[1:]:
    i += 1
    ltype = layer.__class__.__name__
    acts = layer.output
    f = K.function([discriminator.input], [acts])

    acts = f([x_train[:200]])[0]
    acts_np = np.array(acts)
    acts_np = acts_np.reshape((np.prod(acts_np.shape),))
    plt.hist(acts_np, label = "layer_{}_{}_real".format(ltype, i), alpha=0.5, bins=100)

    acts = f([generated_samples[:200]])[0]
    acts_np = np.array(acts)
    acts_np = acts_np.reshape((np.prod(acts_np.shape),))
    plt.hist(acts_np, label = "layer_{}_{}_gen".format(ltype,i), alpha=0.5, bins=100)
    plt.legend()
    fileName = prefix + "_act_hist_layer_{}_{}.png".format(ltype,i)
    print("Creating file: " + fileName)
    plt.savefig(fileName)
    plt.close()


if gendisc:
    # visualize gradient magnitudes
    plt.figure(figsize=(12,12))
    i = 0
    for layer in gendisc.layers[1:]:
        i += 1
        ltype = layer.__class__.__name__
        acts = layer.output
        random = np.random.normal(size=(x_train.shape[0], args.latent_dim))
        f = K.function([gendisc.input], K.gradients(gendisc.output, [layer.output]))

        acts = f([random[:200]])[0]
        print((acts.shape))
        acts_np = np.array(acts)
        acts_np = acts_np.reshape((np.prod(acts_np.shape),))
        plt.hist(acts_np, label = "layer_{}_{}_real".format(ltype, i), alpha=0.5, bins=100)

        plt.legend()
        fileName = prefix + "_grad_hist_layer_{}_{}.png".format(ltype,i)
        print("Creating file: " + fileName)
        plt.savefig(fileName)
        plt.close()
