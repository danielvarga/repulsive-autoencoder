from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import math


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
def displayImageManifold(n, latent_dim, generator, height, width, xdim, ydim, zdim, name):
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
            z_sample = z_sample.reshape([1,latent_dim])
            x_decoded = generator.predict(z_sample)
            image = x_decoded[0].reshape(height, width)
            images_up.append(image)
            z_sample_down = np.array([0] * xdim + [xi] + [0] * (ydim-xdim-1) + [yi] + [0] * (zdim-ydim-1) + [-zi] + [0] * (latent_dim - zdim-1))
            z_sample_down = z_sample_down.reshape([1,latent_dim])
            x_decoded_down = generator.predict(z_sample_down)
            image_down = x_decoded_down[0].reshape(height, width)
            images_down.append(image_down)

    images = np.concatenate([np.array(images_up), np.array(images_down)])
    plotImages(images, n, 2*n, name)


def displayRandom(n, latent_dim, sampler, generator, height, width, name):
    images = []
    for i in range(n):
        for j in range(n):
            z_sample = sampler(1, latent_dim)
            x_decoded = generator.predict(z_sample)
            image = x_decoded[0].reshape(height, width)
            images.append(image)
    plotImages(np.array(images), n, n, name)

