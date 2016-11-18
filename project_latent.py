import matplotlib
import numpy as np
import numpy.linalg
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt

import vis
import data
import model


#encoder = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/vae_baseline_300_encoder")
#generator = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/vae_baseline_300_generator")
#batch_size = 1000

#encoder = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/nvae_baseline_200_encoder")
#generator = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/nvae_baseline_200_generator")
#batch_size = 1000

#modelname = "vae_var_100"
#prefix = "/home/zombori/latent/" + modelname
#encoder = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/" + modelname + "_encoder")
#encoder_var = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/" + modelname + "_encoder_var")
#generator = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/" + modelname + "_generator")
#batch_size = 1000

#encoder = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d2_vae/disc_3_1000_d2_vae_encoder")
#generator = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d2_vae/disc_3_1000_d2_vae_generator")
#batch_size = 250

shape = (72, 60)

# encoder = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/vae_baseline_300_encoder")
# generator = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/vae_baseline_300_generator")
# batch_size = 1000

# encoder = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d2_vae/disc_3_1000_d2_vae_encoder")
# generator = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d2_vae/disc_3_1000_d2_vae_generator")
# batch_size = 250

# encoder = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d3_vae/disc_3_1000_d3_vae_encoder")
# generator = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d3_vae/disc_3_1000_d3_vae_generator")
# batch_size = 250

shape = (72, 64)
encoder = vis.loadModel("/home/csadrian/repulsive-autoencoder/disc_l50_e300_d2_vae_encoder")
generator = vis.loadModel("/home/csadrian/repulsive-autoencoder/disc_l50_e300_d2_vae_generator")
batch_size = 250



(x_train, x_test), (height, width) = data.load("celeba", shape=shape)
latent_train = encoder.predict(x_train, batch_size = batch_size)
latent_test = encoder.predict(x_test, batch_size = batch_size)


do_latent_variances = False
if do_latent_variances:
    latent_train_mean = encoder.predict(x_train, batch_size = batch_size)
    latent_test_mean = encoder.predict(x_test, batch_size = batch_size)

    latent_train_logvar = encoder_var.predict(x_train, batch_size = batch_size)
    latent_test_logvar = encoder_var.predict(x_test, batch_size = batch_size)

    latent_train = np.random.normal(size=latent_train_mean.shape) * np.exp(latent_train_logvar/2) + latent_train_mean
    latent_test = np.random.normal(size=latent_test_mean.shape) * np.exp(latent_test_logvar/2) + latent_test_mean

    np.savez(prefix + "_train_latent_mean.npz", latent_train_mean)
    np.savez(prefix + "_train_latent_logvar.npz", latent_train_logvar)
    np.savez(prefix + "_train_latent.npz", latent_train)

    np.savez(prefix + "_test_latent.npz", latent_test)

variances = np.var(latent_train, axis=0)
working_mask = (variances > 0.2)
print np.sum(working_mask), "/", working_mask.shape
print np.histogram(variances)

n = latent_train.shape[1]

def masked_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    return z * working_mask

vis.displayRandom(n=20, latent_dim=n, sampler=masked_sampler,
        generator=generator, height=72, width=60, name=prefix + "_masked", batch_size=batch_size)


projector = GaussianRandomProjection(n_components=2, random_state=81)
projected_train = projector.fit_transform(latent_train)
projected_test = projector.fit_transform(latent_test)

projected_train = latent_train[:, [0,1]]
projected_test = latent_test[:, [0,1]]

mymin = np.min((np.min(projected_train), np.min(projected_test)))
mymax = np.max((np.max(projected_train), np.max(projected_test)))

plt.figure(figsize=(12,6))
plt.xlim(mymin,mymax)
plt.ylim(mymin,mymax)
ax1 = plt.subplot(121)
ax1.hexbin( projected_train[:, 0], projected_train[:, 1])
plt.xlim(mymin,mymax)
plt.ylim(mymin,mymax)
ax2 = plt.subplot(122)
ax2.hexbin( projected_test[:, 0], projected_test[:, 1])
plt.savefig(prefix + "_hexbin.png")

corr_train = np.corrcoef(latent_train.T)
corr_test = np.corrcoef(latent_test.T)



plt.figure(figsize=(12,24))
ax1 = plt.subplot(211)
ax1.matshow(np.abs(corr_train), cmap='coolwarm')
ax2 = plt.subplot(212)
ax2.matshow(np.abs(corr_test), cmap='coolwarm')
plt.savefig(prefix + "_corr.png")

cov_train = np.cov(latent_train.T)
eigvals = list(np.linalg.eigvals(cov_train).real)
print "cov_train eigvals = ", sorted(eigvals, reverse=True)


print "CS", cov_train.shape
mean_train = np.mean(latent_train, axis=0)
print "MS", mean_train.shape
cho = np.linalg.cholesky(cov_train)
print "CHOS", cho.shape
N = 100000
z = np.random.normal(0.0, 1.0, (N, n))
sample = cho.dot(z.T).T+mean_train
print sample.shape

corr_learned = np.corrcoef(sample.T)
plt.figure(figsize=(12,24))
ax1 = plt.subplot(211)
ax1.matshow(np.abs(corr_train), cmap='coolwarm')
ax2 = plt.subplot(212)
ax2.matshow(np.abs(corr_learned), cmap='coolwarm')
plt.savefig(prefix + "_corr_learned.png")


vis.displayRandom(n=20, latent_dim=n, sampler=model.gaussian_sampler,
        generator=generator, height=72, width=60, name=prefix + "_standard", batch_size=batch_size)

def oval_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    return cho.dot(z.T).T+mean_train

vis.displayRandom(n=20, latent_dim=n, sampler=oval_sampler,
        generator=generator, height=72, width=60, name=prefix + "_oval", batch_size=batch_size)


do_tsne = True

if do_tsne:
    from sklearn.manifold import TSNE
    import sklearn
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, metric="euclidean")
    n = 5000
    latent_train_sampled = latent_train[np.random.choice(latent_train.shape[0], size=n, replace=False)]
    # print latent_train_sampled[0, :]
    # print latent_train_sampled[:, 0]
    reduced = tsne.fit_transform(latent_train_sampled)

    plt.figure(figsize=(12,12))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.savefig(prefix + "_tsne.png")
