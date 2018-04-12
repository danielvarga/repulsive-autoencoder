import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import data
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys


directory = "/home/zombori/latent/"
model = "vae_var_100"
mean_file = directory + model + "_train_latent_mean.npz"
logvar_file = directory + model + "_train_latent_logvar.npz"

mean = np.load(mean_file)["arr_0"]
logvar = np.load(logvar_file)["arr_0"]
var = np.exp(logvar)
center_variances = np.var(mean, axis=0)
working_mask = (center_variances > 0.2)
# print "\n".join(map(str, list(enumerate(working_mask))))

print(mean.shape, var.shape)
assert mean.shape == var.shape
n, d = mean.shape


# Three relevant coords for vae_var_100:
# coords = [9, 45, 209]
# Two relevant and one irrelevant coord:
coords = [9, 45, 10]
mean_3d = mean[:500, coords]
var_3d = var[:500, coords]
sd_3d = np.sqrt(var_3d)

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')
ax.scatter(mean_3d[:, 0], mean_3d[:, 1], mean_3d[:, 2])
for i in range(len(mean_3d)):
    ax.plot([mean_3d[i, 0]-sd_3d[i, 0], mean_3d[i, 0]+sd_3d[i, 0]],
            [mean_3d[i, 1], mean_3d[i, 1]],
            [mean_3d[i, 2], mean_3d[i, 2]], c="blue")
    ax.plot([mean_3d[i, 0], mean_3d[i, 0]],
            [mean_3d[i, 1]-sd_3d[i, 1], mean_3d[i, 1]+sd_3d[i, 1]],
            [mean_3d[i, 2], mean_3d[i, 2]], c="red")
    ax.plot([mean_3d[i, 0], mean_3d[i, 0]],
            [mean_3d[i, 1], mean_3d[i, 1]],
            [mean_3d[i, 2]-sd_3d[i, 2], mean_3d[i, 2]+sd_3d[i, 2]], c="green")

#    ax.plot([mean_3d[i, 0], mean_3d[i, 0]+sd_3d[i, 0]],
#            [mean_3d[i, 1], mean_3d[i, 1]+sd_3d[i, 1]],
#            [mean_3d[i, 2], mean_3d[i, 2]+sd_3d[i, 2]])
ax.set_xlim(-4, +4)
ax.set_ylim(-4, +4)
ax.set_zlim(-4, +4)
plt.show()
sys.exit()

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
