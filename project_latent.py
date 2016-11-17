import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.linalg
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt

import vis
import data
import model


# encoder = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/vae_baseline_300_encoder")
# generator = vis.loadModel("/home/zombori/repulsive-autoencoder/pictures/vae_baseline_300_generator")
encoder = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d2_vae/disc_3_1000_d2_vae_encoder")
generator = vis.loadModel("/home/csadrian/repulsive-autoencoder/models/disc_3_1000_d2_vae/disc_3_1000_d2_vae_generator")


(x_train, x_test), (height, width) = data.load("celeba")
latent_train = encoder.predict(x_train, batch_size = 250)
latent_test = encoder.predict(x_test, batch_size = 250)
np.savez("latent_train.npz", latent_train)
np.savez("latent_test.npz", latent_test)

projector = GaussianRandomProjection(n_components=2, random_state=81)
projected_train = projector.fit_transform(latent_train)
projected_test = projector.fit_transform(latent_test)

#projected_train = latent_train[:, [0,1]]
#projected_test = latent_test[:, [0,1]]
prefix = "project_random"

mymin = np.min((np.min(projected_train), np.min(projected_test)))
mymax = np.max((np.max(projected_train), np.max(projected_test)))

plt.figure(figsize=(12,6))
plt.xlim(mymin,mymax)
plt.ylim(mymin,mymax)
ax1 = plt.subplot(121)
ax1.hexbin( projected_train[:, 0], projected_train[:, 1])
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
plt.savefig(prefix + "_cov.png")

cov_train = np.cov(latent_train.T)
print "CS", cov_train.shape
mean_train = np.mean(latent_train, axis=0)
print "MS", mean_train.shape
cho = np.linalg.cholesky(cov_train)
print "CHOS", cho.shape
n = cho.shape[0]
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
plt.savefig(prefix + "_cov_learned.png")


vis.displayRandom(n=50, latent_dim=n, sampler=model.gaussian_sampler,
        generator=generator, height=72, width=60, name="standard.png", batch_size=250)

def oval_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    return cho.dot(z.T).T+mean_train

vis.displayRandom(n=50, latent_dim=n, sampler=oval_sampler,
        generator=generator, height=72, width=60, name="oval.png", batch_size=250)


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
