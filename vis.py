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
    height, width = data.shape[-2:]
    height_inc = height+1
    width_inc = width+1
    n = len(data)
#    assert n <= n_x*n_y
    if n > n_x*n_y: n = n_x * n_y

    image_data = np.zeros(
        (height_inc * n_y + 1, width_inc * n_x - 1),
        dtype='uint8'
    )
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx].reshape((height, width))
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data)
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


def displayRandom(n, latent_dim, sampler, generator, height, width, name, batch_size=32):
    images = []
    for i in range(n):
        for j in range(n):
            z_sample = sampler(batch_size, latent_dim)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            image = x_decoded[0].reshape(height, width)
            images.append(image)
    plotImages(np.array(images), n, n, name)

def displaySet(imageBatch, n, generator, height, width, name):
    batchSize = imageBatch.shape[0]
    nsqrt = int(np.ceil(np.sqrt(n)))
    recons = generator.predict(imageBatch, batch_size=batchSize)

    mergedSet = np.zeros(shape=[n*2] + list(imageBatch.shape[1:]))
    for i in range(n):
        mergedSet[2*i] = imageBatch[i]
        mergedSet[2*i+1] = recons[i]
    result = mergedSet.reshape([2*n,height,width])
    plotImages(result, 2*nsqrt, nsqrt, name)

def displayInterp(x_train, x_test, batch_size, dim, height, width, encoder, generator, gridSize, name):
    train_latent = encoder.predict(x_train[:batch_size], batch_size=batch_size)
    test_latent = encoder.predict(x_test[:batch_size], batch_size=batch_size)
    parallelogram_point = test_latent[0] + train_latent[1] - train_latent[0]
    anchors = np.array([train_latent[0], train_latent[1], test_latent[0], parallelogram_point])
    interpGrid = grid_layout.create_mine_grid(gridSize, gridSize, dim, gridSize-1, anchors, True, False) # TODO different interpolations for different autoencoders!!!
    n = interpGrid.shape[0]
    interpGrid = np.repeat(interpGrid, (batch_size//n) + 1, axis=0)[0:batch_size]
    predictedGrid = generator.predict(interpGrid, batch_size=batch_size)
    predictedGrid = predictedGrid[0:n]

    prologGrid = np.zeros([gridSize,height*width])
    prologGrid[0:3] = [x_train[0],x_train[1], x_test[0]]

    grid = np.concatenate([prologGrid, predictedGrid])
    reshapedGrid = grid.reshape([grid.shape[0], height, width])
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
    
