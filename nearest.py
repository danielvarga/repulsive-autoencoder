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
if data_object.finite:
    x_test = data_object.get_finite_set()
elif data_object.synthetic:
    x_test = data_object.get_uniform_data()
else:
    x_train, x_test = data_object.get_data(trainSize, testSize)

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

for g in x_generated:
    nearest_index = t.get_nns_by_vector(g, 1)[0]
    hist[nearest_index] += 1

plt.plot(hist)
fileName = generated.split('.')[0] + "_success.png"
print "Saving histogram to {}".format(fileName)
plt.savefig(fileName)

sorter = reversed(np.argsort(hist)[-20:])
popular_images = []
for i in sorter:
    popular_images.append(x_test[i])
popular_images = np.array(popular_images)
fileName = generated.split('.')[0] + "_popular"
vis.plotImages(popular_images, 20, 1, fileName)
