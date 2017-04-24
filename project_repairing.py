import numpy as np

import samplers
import kohonen


sampler = samplers.spherical_sampler


def averageDistance(dataBatch, fakeBatch):
    return np.mean(np.linalg.norm(dataBatch - fakeBatch, axis=1))


def main():
    latent_dim = 10
    N = 10000
    n = 100
    epoch_count = 1000
    minibatchCount = N // n
    data = sampler(N, latent_dim)
    masterPermutation = np.random.permutation(N)

    for epoch_index in range(epoch_count):
        totalMeanDistance = averageDistance(data[::10], data[masterPermutation][::10])
        print "totalMeanDistance", totalMeanDistance,
        epochDistances = []
        fixedPointRatios = []
        allIndices = np.random.permutation(N)
        for i in range(minibatchCount):
            dataIndices = allIndices[i*n:(i+1)*n]
            dataBatch = data[dataIndices]
            latentBatch = data[masterPermutation[dataIndices]]
            batchPermutation = np.array(kohonen.optimalPairing(latentBatch, dataBatch))
            masterPermutation[dataIndices] = masterPermutation[dataIndices[batchPermutation]]
            fixedPointRatio = float(np.sum(batchPermutation == np.arange(n))) / n
            minibatchDistances = averageDistance(dataBatch, latentBatch)
            fixedPointRatios.append(fixedPointRatio)
            epochDistances.append(minibatchDistances)

        epochInterimMean = np.mean(np.array(epochDistances))
        epochFixedPointRatio = np.mean(np.array(fixedPointRatios))
        print epochInterimMean, epochFixedPointRatio

main()
