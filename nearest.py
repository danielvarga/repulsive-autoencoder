import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import annoy
from PIL import Image
import matplotlib.pyplot as plt

import data
import vis

dataset, generated = sys.argv[1:]
generated_images = np.load(generated)

shape = generated_images.shape[1:3] # only tensorflow
testSize = 100
trainSize = 1
if generated_images.shape[3] == 3:
    color = True
else:
    color = False

data_object = data.load(dataset, shape, color)
if data_object.synthetic:
    x_test = data_object.get_uniform_data()
else:
    x_train, x_test = data_object.get_data(trainSize, testSize)

if False:
    m, mprime, l = data_object.get_M_Mprime_L(generated_images)
    print(m, mprime, l)
    emd = vis.dataset_emd(x_test, generated_images[:1000])
    print(m, mprime, l, emd)
    xxx

x_generated = generated_images.reshape(generated_images.shape[0], -1)
x_true = x_test.reshape(x_test.shape[0], -1)
f = x_true.shape[1]
t = annoy.AnnoyIndex(f, metric="euclidean")

for i, v in enumerate(x_true):
    t.add_item(i, v)

sys.stderr.write("items added\n")

t.build(100)

sys.stderr.write("tree built\n")

hist = np.zeros(len(x_true))
nearest_points = []
for g in x_generated:
    nearest_index = t.get_nns_by_vector(g, 1)[0]
    hist[nearest_index] += 1
    nearest_points.append(x_true[nearest_index])

plt.plot(hist)
prefix = generated.split('.')[0]
fileName = prefix + "_success.png"
print("Saving histogram to {}".format(fileName))
plt.savefig(fileName)

nearest_points = np.array(nearest_points)
mu = np.mean(nearest_points)
sigma = np.std(nearest_points)
print("mu: {}, sigma: {}".format(mu, sigma))
vis.cumulative_view(nearest_points, "Nearest points cdf", prefix + "_cdf.png")

sorter = reversed(np.argsort(hist)[-20:])
popular_images = []
for i in sorter:
    popular_images.append(x_test[i])
popular_images = np.array(popular_images)
fileName = generated.split('.')[0] + "_popular"
vis.plotImages(popular_images, 20, 1, fileName)
