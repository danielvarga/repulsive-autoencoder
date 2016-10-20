from PIL import Image
import numpy as np
import math

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
# !!! only works if xdim < ydim < zdim TODO
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
            # Padded with zeroes at the rest of the coordinates:
            z_sample = np.array([0] * xdim + [xi] + [0] * (ydim-xdim-1) + [yi] + [0] * (zdim-ydim-xdim-2) + [zi] + [0] * (latent_dim - zdim-ydim-xdim-3))
            z_sample = z_sample.reshape([1,latent_dim])
            x_decoded = generator.predict(z_sample)
            image = x_decoded[0].reshape(height, width)
            images_up.append(image)
            z_sample_down = np.array([0] * xdim + [xi] + [0] * (ydim-xdim-1) + [yi] + [0] * (zdim-ydim-xdim-2) + [-zi] + [0] * (latent_dim - zdim-ydim-xdim-3))
            z_sample_down = z_sample_down.reshape([1,latent_dim])
            x_decoded_down = generator.predict(z_sample_down)
            image_down = x_decoded_down[0].reshape(height, width)
            images_down.append(image_down)

    images = np.concatenate([np.array(images_up), np.array(images_down)])
    plotImages(images, n, 2*n, name)


def displayRandom(n, latent_dim, generator, height, width, name):
    images = []
    for i in range(n):
        for j in range(n):
            z_sample = np.random.normal(size=(1, latent_dim))
            z_sample /= np.linalg.norm(z_sample)
            x_decoded = generator.predict(z_sample)
            image = x_decoded[0].reshape(height, width)
            images.append(image)
    plotImages(np.array(images), n, n, name)

