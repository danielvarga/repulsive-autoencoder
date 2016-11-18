import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import data
from matplotlib import cm

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

fig, ax = plt.subplots()

if ax is None:
    ax = plt.gca()

c1 = np.sum(working_mask * var, axis=1) / np.sum(working_mask)
c2 = np.sum(irrelevant_variance, axis=1) / np.sum(1-working_mask)

(x_train, x_test), (height, width) = data.load("celeba")


i = 0
for x, y in zip(c1, c2):
    i = i + 1
    if i > 300:
	continue
    im_a = x_train[i-1].reshape(72, 60)
    image = Image.fromarray(im_a)
    im = OffsetImage(image, zoom=0.5, cmap=cm.gray)
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    ax.add_artist(ab)

ax.autoscale()
plt.scatter(c1, c2)


plt.show()

#quit()

plt.savefig("relevant_var_irrelevant_var.png")


center_score = np.sqrt(np.sum(mean ** 2, axis=1))
var_score = np.sqrt(np.sum(irrelevant_variance, axis=1))
assert center_score.shape==(n,)
assert var_score.shape==(n,)
plt.scatter(center_score, var_score)
plt.savefig("relevant_mean_irrelevant_sd.png")
