import matplotlib
matplotlib.use('Agg')
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import math

import grid_layout
from keras.models import model_from_json


# TODO Add optional arg y_test for labeling.
def latentScatter(encoder, x_test, batch_size, name):
    # # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    points = x_test_encoded.copy()
    points[:, 0] += 2.2 * (points[:, 2]>=0) # TODO This badly fails for normal latent vars.
    plt.scatter(points[:, 0], points[:, 1])
    fileName = name + ".png"
    print "Creating file " + fileName
    plt.savefig(fileName)


def plotImages(data, n_x, n_y, name):
    (height, width, channel) = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    if channel == 1:
        mode = "L"
        data = data[:,:,:,0]
        image_data = np.zeros((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    else:
        mode = "RGB"
        image_data = np.zeros((height_inc * n_y + 1, width_inc * n_x - 1, channel), dtype='uint8')
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)
    fileName = name + ".png"
    print "Creating file " + fileName
    img.save(fileName)

# display a 2D manifold of the images
# TODO Only works for spherical distributions.
#      More precisely, it works for normals, but is very misleading.
# TODO only works if xdim < ydim < zdim
def displayImageManifold(n, latent_dim, generator, height, width, xdim, ydim, zdim, name, batch_size=32):
    grid_x = np.linspace(-1, +1, n)
    grid_y = np.linspace(-1, +1, n)

    images_up=[]
    images_down=[]
    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):
            zisqr = 1.000001-xi*xi-yi*yi
            if zisqr < 0.0:
                images_up.append(np.zeros([height,width]))
                images_down.append(np.zeros([height,width]))
                continue
            zi = math.sqrt(zisqr)
            z_sample = np.array([0] * xdim + [xi] + [0] * (ydim-xdim-1) + [yi] + [0] * (zdim-ydim-1) + [zi] + [0] * (latent_dim - zdim-1))
            z_sample = z_sample.reshape([batch_size, latent_dim])
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            image = x_decoded[0].reshape(height, width)
            images_up.append(image)
            z_sample_down = np.array([0] * xdim + [xi] + [0] * (ydim-xdim-1) + [yi] + [0] * (zdim-ydim-1) + [-zi] + [0] * (latent_dim - zdim-1))
            z_sample_down = z_sample_down.reshape([batch_size, latent_dim])
            x_decoded_down = generator.predict(z_sample_down, batch_size=batch_size)
            image_down = x_decoded_down[0].reshape(height, width)
            images_down.append(image_down)

    images = np.concatenate([np.array(images_up), np.array(images_down)])
    plotImages(images, n, 2*n, name)


def displayPlane(x_train, latent_dim, plane, generator, name, batch_size=32, showNearest=False):
    images = []
    height, width, l_d = plane.shape
    assert l_d == latent_dim
    cnt = height * width
    cnt_aligned = (cnt // batch_size + 1) * batch_size
    plane_flat = plane.reshape((cnt, latent_dim))
    plane_tailed = np.copy(plane_flat)
    plane_tailed.resize((cnt_aligned, latent_dim)) # Extra zeros added.
    x_decoded = generator.predict(plane_tailed, batch_size=batch_size)
    x_decoded = x_decoded[:cnt] # Extra zeros removed.
    shape = [cnt] + list(x_train.shape[1:])
    images = x_decoded.reshape(shape)
    if not showNearest:
        plotImages(images, width, height, name)
    else:
        distToTrain = distanceMatrix(images.reshape([images.shape[0],-1]), x_train.reshape([x_train.shape[0],-1]))
        distIndices = distToTrain.argmin(axis=0)
        nearestTrain = x_train[distIndices]
        images2 = []
        for i in range(images.shape[0]):
            images2.append(images[i])
            images2.append(nearestTrain[i])
        plotImages(np.array(images2), 2*width, height, name)


def displayRandom(n, x_train, latent_dim, sampler, generator, name, batch_size=32, showNearest=False):
    images = []
    cnt = n * n
    cnt_aligned = (cnt // batch_size + 1) * batch_size
    z_sample = sampler(cnt_aligned, latent_dim)
    x_decoded = generator.predict(z_sample, batch_size=batch_size)
    x_decoded = x_decoded[:cnt]
    indx = 0
    for i in range(n):
        for j in range(n):
            image = x_decoded[indx].reshape(x_train.shape[1:])
            images.append(image)
            indx += 1
    assert indx == cnt
    images = np.array(images)
    if not showNearest:
        plotImages(np.array(images), n, n, name)
    else:
        distToTrain = distanceMatrix(images.reshape([images.shape[0],-1]), x_train.reshape([x_train.shape[0],-1]))
        distIndices = distToTrain.argmin(axis=0)
        nearestTrain = x_train[distIndices]
        images2 = []
        for i in range(images.shape[0]):
            images2.append(images[i])
            images2.append(nearestTrain[i])
        plotImages(np.array(images2), 2*n, n, name)

def displaySet(imageBatch, n, generator, name):
    batchSize = imageBatch.shape[0]
    nsqrt = int(np.ceil(np.sqrt(n)))
    recons = generator.predict(imageBatch, batch_size=batchSize)

    mergedSet = np.zeros(shape=[n*2] + list(imageBatch.shape[1:]))
    for i in range(n):
        mergedSet[2*i] = imageBatch[i]
        mergedSet[2*i+1] = recons[i]
    result = mergedSet.reshape([2*n] + list(imageBatch.shape[1:]))
    plotImages(result, 2*nsqrt, nsqrt, name)

def displayInterp(x_train, x_test, batch_size, dim, encoder, generator, gridSize, name):
    train_latent = encoder.predict(x_train[:batch_size], batch_size=batch_size)
    test_latent = encoder.predict(x_test[:batch_size], batch_size=batch_size)
    parallelogram_point = test_latent[0] + train_latent[1] - train_latent[0]
    anchors = np.array([train_latent[0], train_latent[1], test_latent[0], parallelogram_point])
    interpGrid = grid_layout.create_mine_grid(gridSize, gridSize, dim, gridSize-1, anchors, True, False) # TODO different interpolations for different autoencoders!!!
    n = interpGrid.shape[0]
    interpGrid = np.repeat(interpGrid, (batch_size//n) + 1, axis=0)[0:batch_size]
    predictedGrid = generator.predict(interpGrid, batch_size=batch_size)
    predictedGrid = predictedGrid[0:n]

    prologGrid = np.zeros([gridSize] + list(x_train.shape[1:]))
    prologGrid[0:3] = [x_train[0],x_train[1], x_test[0]]

    grid = np.concatenate([prologGrid, predictedGrid])
    reshapedGrid = grid.reshape([grid.shape[0]] + list(x_train.shape[1:]))
    plotImages(reshapedGrid, gridSize, gridSize, name)

def saveModel(model, filePrefix):
    jsonFile = filePrefix + ".json"
    weightFile = filePrefix + ".h5"
    with open(filePrefix + ".json", "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(weightFile)
    print "Saved model to files {}, {}".format(jsonFile, weightFile)

def loadModel(filePrefix):
    jsonFile = filePrefix + ".json"
    weightFile = filePrefix + ".h5"
    jFile = open(jsonFile, 'r')
    loaded_model_json = jFile.read()
    jFile.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weightFile)
    print "Loaded model from files {}, {}".format(jsonFile, weightFile)
    return model
    

# returns matrix M, such that M[i][j] is the distance between the j-th row in x and the i-th row in y
def distanceMatrix(x, y):
    xL2S = np.sum(x*x, axis=-1)
    yL2S = np.sum(y*y, axis=-1)
    xL2SM = np.tile(xL2S, (len(y), 1))
    yL2SM = np.tile(yL2S, (len(x), 1))
    squaredDistances = xL2SM + yL2SM.T - 2.0*y.dot(x.T)
    distances = np.sqrt(squaredDistances+1e-6) # elementwise. +1e-6 is to supress sqrt-of-negative warning.
    return distances

def plotMVVM(x_train, encoder, encoder_var, batch_size, name):
    latent_train_mean = encoder.predict(x_train, batch_size = batch_size)
    mean_variances = np.var(latent_train_mean, axis=0)
    latent_train_logvar = encoder_var.predict(x_train, batch_size = batch_size)
    variance_means = np.mean(np.exp(latent_train_logvar), axis=0)
    xlim = (-1, 12)
    ylim = (-1, 3)
    plt.figure(figsize=(12,6))
    plt.scatter(mean_variances, variance_means)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    print "Creating file " + name
    plt.savefig(name)

def plotMVhist(x_train, encoder, batch_size, name):
    latent_train_mean = encoder.predict(x_train, batch_size = batch_size)
    mean_variances = np.var(latent_train_mean, axis=0)
    histogram = np.histogram(mean_variances, 30)
    mean_variances = histogram[1]
    variance_means = [0] + list(histogram[0])
    xlim = (0,np.max(mean_variances))
    ylim = (0,np.max(variance_means))
    plt.figure(figsize=(12,6))
    plt.scatter(mean_variances, variance_means)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    print "Creating file " + name
    plt.savefig(name)

def plot2Dprojections(dataset, indices, name):
    dims = len(indices)
    plt.figure(figsize=(6 * dims, 6 * dims))
    f, axarr = plt.subplots(dims, dims)
    for i in range(dims):
        for j in range(i,dims):
            axarr[i, j].hexbin(dataset[:, indices[i]], dataset[:, indices[j]])
            plt.xlim(4, 4)
            plt.ylim(4, 4)
        print i
    print "Creating file " + name
    plt.savefig(name)
    plt.close()

"""
def edgeDetect(images):
    (height, width, channel) = images.shape[1:]
    horizontalEdges = np.zeros(images.shape)
    horizontalEdges[:,:height-1,:,:] = images[:,:height-1,:,:] - images[:,1:,:,:]
    verticalEdges = np.zeros(images.shape)
    verticalEdges[:,:,:width-1:,:] = images[:,:,:width-1,:] - images[:,:,1:,:]    
    diagonalEdges = np.zeros(images.shape)
    diagonalEdges[:,:height-1,:width-1,:] = images[:,:height-1,:width-1,:] - images[:,1:,1:,:]
    edges = horizontalEdges + verticalEdges + diagonalEdges
    return edges
"""
