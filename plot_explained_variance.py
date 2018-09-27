from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import sklearn.decomposition

import numpy as np
import os


# prefix = "/home/daniel/experiments/repulsive-autoencoder/pictures/dcgan_vae_lsun/dcgan_vae_lsun_latent"
# prefix = "/mnt/g2home/daniel/experiments/repulsive-autoencoder/pictures/dcgan_vae_lsun/dcgan_vae_lsun_latent"
# prefix = "/home/zombori/experiments/repulsive-autoencoder/pictures/dcgan_vae_largelatent/dcgan_vae_largelatent_latent"
# prefix = "/home/csadrian/ra-new/repulsive-autoencoder/pictures/dcgan_vae_lsun_newkl_grad_from_zero/dcgan_vae_lsun_newkl_grad_from_zero_latent"
prefixes = []
prefixes.append({"prefix": "vae_resnet", "label": "Baseline VAE"})
prefixes.append({"prefix": "vae_resnet_cov_10000", "label": "Baseline VAE + $\lambda_{cov}=10000$"})
prefixes.append({"prefix": "vae_resnet_newkl", "label": "$\lambda_{newkl}=1$"})
prefixes.append({"prefix": "vae_resnet_newkl_10", "label": "$\lambda_{newkl}(1)=10$"})
prefixes.append({"prefix": "vae_resnet_newkl_10_cov", "label": "$\lambda_{newkl}(1)=10, \lambda_{cov}=1$"})
prefixes.append({"prefix": "vae_resnet_newkl_10_cov_10", "label": "$\lambda_{newkl}(1)=10, \lambda_{cov}=10$"})
prefixes.append({"prefix": "vae_resnet_newkl_10_cov_100", "label": "$\lambda_{newkl}(1)=10, \lambda_{cov}=100$"})
prefixes.append({"prefix": "vae_resnet_newkl_10_cov_1000", "label": "$\lambda_{newkl}(1)=10, \lambda_{cov}=1000$"})
prefixes.append({"prefix": "vae_resnet_newkl_10_cov_10000", "label": "$\lambda_{newkl}(1)=10, \lambda_{cov}=10000$"})


for data in prefixes:

  prefix = data['prefix']
  file_prefix = "pictures" + "/" + prefix + "/" + prefix + "_latent"

  meanFile = file_prefix + "_mean_200.npy"
  logvarFile = file_prefix + "_log_var_200.npy"

  mean = np.load(meanFile)
  logvar = np.load(logvarFile)
  var = np.exp(logvar)
  std = np.exp(logvar/2)

  print("Mean shape: ", mean.shape)
  print("Var shape: ", var.shape)

  latent_dim = mean.shape[1]

  pca = sklearn.decomposition.PCA()
  pca.fit_transform(mean)

  plt.plot(pca.explained_variance_ratio_.cumsum(), label=prefix)

plt.title("Explained variance with the principal components of latent means \n with the new KL loss and identity covariance loss")
plt.legend(
  [l['label'] for l in prefixes]
  )
plt.ylim(-0.01, 1.01)
name = "variance_explained_plot.png"
print("Creating file " + name)
plt.savefig(name)
plt.close()

