import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm
from PIL import Image
import os
import re
import numpy as np
from sklearn.manifold import TSNE
import sklearn

import vis
import data
import wgan_params
import kohonen

args = wgan_params.getArgs()
print(args)
prefix = args.prefix

# set random seed
np.random.seed(10)

prefix = args.prefix
generator_prefix = args.prefix + "_generator"
generator = vis.loadModel(generator_prefix)

def biflatten(x):
    assert len(x.shape)>=2
    return x.reshape(x.shape[0], -1)

def averageDistance(dataBatch, fakeBatch):
    return np.mean(np.linalg.norm(biflatten(dataBatch) - biflatten(fakeBatch), axis=1))

print "loading data"
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)

print "loading latent points"
latent_file = prefix + "_latent.npy"
latent = np.load(latent_file)
fake_original = generator.predict(latent, batch_size=args.batch_size)

if True:
    print "Checking covariance matrix of latent points"
    cov_latent = np.cov(latent.T)
    eigVals, eigVects = np.linalg.eigh(cov_latent)
    print "cov_latent eigvals = ", list(reversed(eigVals))

if False:
    print "displaying images along latent axes"
    ls = []
    steps = np.linspace(-1, 1, num=args.batch_size)
    for dim in range(args.latent_dim):
        l = np.zeros((args.batch_size, args.latent_dim))
        l[:,dim] = steps
        ls.append(l)
    ls = np.concatenate(ls)
    images = generator.predict(ls, batch_size=args.batch_size)
    vis.plotImages(images, args.batch_size, args.latent_dim, prefix + "_latent_axes")
    
if True:
    print "displaying latent structure projected to 2 dimensions"
    tsne_points = 2000
    display_points = 150
    images = fake_original[:tsne_points]
    images *= 255
    images = np.clip(images, 0, 255).astype('uint8')

    fig, ax = plt.subplots()
    if ax is None:
        ax = plt.gca()

    tsne = TSNE(n_components=2, random_state=42, perplexity=100, metric="euclidean")
    reduced = tsne.fit_transform(latent[:tsne_points])
    for i in range(display_points):
        x = reduced[i, 0]
        y = reduced[i, 1]
        im_a = images[i]
        if im_a.shape[2] == 1:
            im_a = np.concatenate([im_a, im_a, im_a], axis=2)
            image = Image.fromarray(im_a, mode="RGB")
#            image = Image.fromarray(im_a[:,:,0], mode="L")
        else:
            image = Image.fromarray(im_a, mode="RGB")
        im = OffsetImage(image, zoom=0.5)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)

    plt.scatter(reduced[:, 0], reduced[:, 1])
    file = prefix + "_latent_structure.png"
    print "Saving {}".format(file)
    plt.savefig(file)
    plt.close()

xxx
 


recons = averageDistance(x_train, fake_original)
print "Reconstruction loss: {}".format(recons)

assert latent.shape[1] == args.latent_dim
for dim in range(args.latent_dim):
    l = latent.copy()
    l[:,dim] = np.mean(l[:,dim])
    l /= np.linalg.norm(l, axis=1, keepdims=True)
    fake = generator.predict(l, batch_size=args.batch_size)
    recons_to_orig = averageDistance(x_train, fake)
    recons_to_recons = averageDistance(fake_original, fake)
    print "Reconstruction without {}th dim: {} to x_train, {} to original reconstruction".format(dim, recons_to_orig, recons_to_recons)


xxx

fake_files_string = os.popen("ls {}_fake_*.npy".format(prefix)).read()
fake_files = fake_files_string.strip().split("\n")
fake_epochs = []
for fake_file in fake_files:
    epoch = int(re.search(".*_fake_(.*).npy", fake_file).group(1))
    fake_epochs.append(epoch)
sorter = np.argsort(fake_epochs)
fake_epochs = [fake_epochs[i] for i in sorter]
fake_files = [fake_files[i] for i in sorter]

fake_images = []
for fake_file in fake_files:
    data = np.load(fake_file)
    assert len(data) == args.trainSize
    fake_images.append(data)
fake_images = np.array(fake_images)

x_train = x_train.reshape((x_train.shape[0], -1))
fake_images = fake_images.reshape((fake_images.shape[0], fake_images.shape[1], -1))

point_limit = 1000
if args.trainSize > point_limit:
    print "Restricting analysis to the first {} points".format(point_limit)
    x_train = x_train[:point_limit]
    fake_images = fake_images[:,:point_limit]

print "Calculating distance matrix"
distances = []
distances10 = []
for i in range(len(fake_epochs)):
    distances.append(kohonen.distanceMatrix(fake_images[i], x_train))
    distances10.append(kohonen.distanceMatrix(fake_images[i], x_train, projection=10))
distances = np.array(distances)
distances10 = np.array(distances)

print "Finding greedy matching"
greedy_matching = np.zeros(len(fake_epochs))
greedy_matching10 = np.zeros(len(fake_epochs))
for i in range(len(fake_epochs)):
    p = np.array(kohonen.greedyPairing(fake_images[i], x_train, distances[i]))
    p10 = np.array(kohonen.greedyPairing(fake_images[i], x_train, distances10[i]))
    greedy_matching[i] = float(np.sum(p == np.arange(len(p)))) / len(p)
    greedy_matching10[i] = float(np.sum(p10 == np.arange(len(p10)))) / len(p10)
print "Fixed point ratio when using greedy matching: ", greedy_matching
print "Fixed point ratio when using greedy matching and 10 dim projection: ", greedy_matching10


print "Finding optimal matching"
optimal_limit = 1000
if args.trainSize > optimal_limit:
    print "Restricting optimal matching to the first {} points".format(optimal_limit)
    x_train = x_train[:optimal_limit]
    fake_images = fake_images[:,:optimal_limit]

optimal_matching = np.zeros(len(fake_epochs))
optimal_matching10 = np.zeros(len(fake_epochs))
for i in range(len(fake_epochs)):
    p = np.array(kohonen.optimalPairing(fake_images[i], x_train, distances[i]))
    p10 = np.array(kohonen.optimalPairing(fake_images[i], x_train, distances10[i]))
    optimal_matching[i] = float(np.sum(p == np.arange(len(p)))) / len(p)
    optimal_matching10[i] = float(np.sum(p10 == np.arange(len(p10)))) / len(p10)
print "Fixed point ratio when using optimal matching: ", optimal_matching
print "Fixed point ratio when using optimal matching and 10 dim projection: ", optimal_matching10
