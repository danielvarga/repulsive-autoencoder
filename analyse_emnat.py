import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import vis
import os
import re
import numpy as np

import data
import wgan_params
import kohonen

args = wgan_params.getArgs()
print(args)
prefix = args.prefix

# set random seed
np.random.seed(10)


fake_files_string = os.popen("ls {}_fake_*.npy".format(prefix)).read()
fake_files = fake_files_string.strip().split("\n")
fake_epochs = []
for fake_file in fake_files:
    epoch = int(re.search(".*_fake_(.*).npy", fake_file).group(1))
    fake_epochs.append(epoch)
sorter = np.argsort(fake_epochs)
fake_epochs = [fake_epochs[i] for i in sorter]
fake_files = [fake_files[i] for i in sorter]

print "loading data"
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)

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
