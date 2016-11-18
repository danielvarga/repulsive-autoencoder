import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

directory = "/home/zombori/latent/"
model = "vae_var_100"
mean_file = directory + model + "_train_latent_mean.npz"
logvar_file = directory + model + "_train_latent_logvar.npz"

mean = np.load(mean_file)["arr_0"]
logvar = np.load(logvar_file)["arr_0"]
var = np.exp(logvar)
assert mean.shape == var.shape
n, d = mean.shape

center_variances = np.var(mean, axis=0)
working_mask = (center_variances > 0.2)

relevant_mean = working_mask * mean
irrelevant_variance = (1 - working_mask) * var

c1 = np.sum(working_mask * var, axis=1) / np.sum(working_mask)
c2 = np.sum(irrelevant_variance, axis=1) / np.sum(1-working_mask)
plt.scatter(c1, c2)
plt.savefig("relevant_var_irrelevant_var.png")


center_score = np.sqrt(np.sum(mean ** 2, axis=1))
var_score = np.sqrt(np.sum(irrelevant_variance, axis=1))
assert center_score.shape==(n,)
assert var_score.shape==(n,)
plt.scatter(center_score, var_score)
plt.savefig("relevant_mean_irrelevant_sd.png")
