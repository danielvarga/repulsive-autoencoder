import sys
import numpy as np

from keras.datasets import mnist as keras_mnist

import kohonen


def biflatten(x):
    assert len(x.shape)>=2
    return x.reshape(x.shape[0], -1)


def averageDistance(dataBatch, fakeBatch):
    return np.mean(np.linalg.norm(biflatten(dataBatch) - biflatten(fakeBatch), axis=1))


def main():
    (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x = x_train[:, :, :, np.newaxis]
    print x.shape

    filename = "pictures/earthmover_mnist_exp_fake_200.npy"
    y = np.load(filename)
    print y.shape

    assert len(x)==len(y)
    data_count = len(x)

    nb_epoch = 1000
    midibatchSize = 10000
    masterPermutation = np.random.permutation(data_count) # np.arange(data_count).astype(np.int32)
    midibatchCount = data_count // midibatchSize

    for i in range(nb_epoch):
        allIndices = np.random.permutation(data_count)
        totalDistance = 0.0
        for j in range(midibatchCount):
            dataMidiIndices = allIndices[j*midibatchSize:(j+1)*midibatchSize]
            assert midibatchSize==len(dataMidiIndices)
            dataMidibatch = x[dataMidiIndices]
            fakeMidibatch = y[masterPermutation[dataMidiIndices]]
            distanceMatrix = kohonen.distanceMatrix(biflatten(fakeMidibatch), biflatten(dataMidibatch), projection=50)

            exactMatching = False
            if exactMatching:
                midibatchPermutation = np.array(kohonen.optimalPairing(biflatten(fakeMidibatch), biflatten(dataMidibatch), distanceMatrix))
            else:
                midibatchPermutation = np.array(kohonen.greedyPairing(biflatten(fakeMidibatch), biflatten(dataMidibatch), distanceMatrix))

            projectedTotalMidibatchDistance = distanceMatrix[range(len(fakeMidibatch)), midibatchPermutation].sum()

            masterPermutation[dataMidiIndices] = masterPermutation[dataMidiIndices[midibatchPermutation]]

            fakeMidibatch = y[masterPermutation[dataMidiIndices]] # recalc
            totalMidibatchDistance = averageDistance(dataMidibatch, fakeMidibatch) * len(dataMidibatch)
            totalDistance += totalMidibatchDistance

            fixedPointRatio = float(np.sum(midibatchPermutation == np.arange(midibatchPermutation.shape[0]))) / midibatchPermutation.shape[0]
            print i, j, fixedPointRatio, projectedTotalMidibatchDistance / midibatchSize, totalMidibatchDistance / midibatchSize
            sys.stdout.flush()
        globalFixedPointRatio = float(np.sum(masterPermutation == np.arange(data_count))) / data_count

        print i, globalFixedPointRatio, totalDistance / data_count


main()
