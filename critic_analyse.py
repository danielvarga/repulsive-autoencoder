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

plt.figure(figsize=(12,12))
plt.hist(positive, label='images', alpha=0.5)
plt.hist(noise, label='noise', alpha=0.5)
plt.hist(generated, label='generated', alpha=0.5)
# for i in range(3):
#     positive_noise = discriminator.predict(x_train + (i+1) / 3.0 * noise_samples, batch_size=batch_size)
#     plt.hist(positive_noise, label="pos_noise_{}".format(i+1))

plt.legend()
plt.savefig(prefix + "_posneg.png")
plt.close()
