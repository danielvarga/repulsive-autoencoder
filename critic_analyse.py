import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import data
import vis

batch_size = 200
latent_dim = 200
train_size = 1000
img_height = 64
img_width = 64

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('--prefix', dest='prefix', type=str, help='Prefix for the saved results.')
parser.add_argument('--discriminator_prefix', dest='discriminator_prefix', type=str, help='Prefix for the saved discriminator.')
parser.add_argument('--generator_prefix', dest='generator_prefix', type=str, help='Prefix for the saved generator.')

args = parser.parse_args()
prefix = args.prefix
discriminator_prefix = args.discriminator_prefix
generator_prefix = args.generator_prefix

discriminator = vis.loadModel(discriminator_prefix)
generator = vis.loadModel(generator_prefix)



(x_train, x_test) = data.load("celeba", train_size, batch_size, shape=(img_height, img_width), color=True)
noise_samples = np.random.uniform(size=x_train.shape)
generated_samples = generator.predict(np.random.normal(size=(x_train.shape[0], latent_dim)), batch_size=batch_size)
positive = discriminator.predict(x_train, batch_size=batch_size)
noise = discriminator.predict(noise_samples, batch_size=batch_size)
generated = discriminator.predict(generated_samples, batch_size=batch_size)

# compare the discriminator with respect to the real images, generated images, random noise
plt.figure(figsize=(12,12))
plt.hist(positive, label='images', alpha=0.5, bins=100)
plt.hist(noise, label='noise', alpha=0.5, bins=100)
plt.hist(generated, label='generated', alpha=0.5, bins=100)
plt.legend()
plt.savefig(prefix + "_posneg.png")
plt.close()

# sort train and generated images according to their discriminator ranking
positive_sorter = np.argsort(np.concatenate([positive[:,0], generated[:,0]]))
x_train_sorted = np.concatenate([x_train, generated_samples])[positive_sorter]
vis.plotImages(x_train_sorted[::5], 20, 20, prefix + "_discriminator_order")

# check what happens if we transform the images
x_train_upside_down = x_train[:,::-1]
x_train_inverted = 1-x_train
upside_down = discriminator.predict(x_train_upside_down, batch_size=batch_size)
inverted = discriminator.predict(x_train_inverted, batch_size=batch_size)
plt.figure(figsize=(12,12))
plt.hist(positive, label='images', alpha=0.5, bins=100)
plt.hist(upside_down, label='upside_down', alpha=0.5, bins=100)
plt.hist(inverted, label='inverted', alpha=0.5, bins=100)
plt.legend()
plt.savefig(prefix + "_transformed.png")
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
plt.savefig(prefix + "_conv_weight_hist.png")    
plt.close()

plt.figure(figsize=(12,12))
bn_layers = [4, 7, 10]
for i in bn_layers:
    weights = discriminator.layers[i].get_weights()
    w = np.array(weights)
    w = w.reshape((np.prod(w.shape),))
    plt.hist(w, label = "bn_layer_{}".format(i), alpha=0.5, bins=100)
plt.legend()
plt.savefig(prefix + "_bn_weight_hist.png")    
plt.close()
