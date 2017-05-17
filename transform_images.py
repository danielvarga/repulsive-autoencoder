import numpy as np

import data
import vis

transformation_types = 3

def test():
    data_object = data.load("mnist", shape=(28, 28))
    x_train, x_test = data_object.get_data(20, 1)
    offsets = np.random.normal(size=(len(x_train), transformation_types))
    transformed = shift(x_train, offsets)
    images = np.concatenate([x_train, transformed])
    vis.plotImages(images, len(x_train), 2, "pictures/transform_images_test")

def shift(images, offsets):
    assert offsets.shape == (len(images), transformation_types)
    transformed = np.zeros(shape=images.shape)
    for i in range(len(images)):
        offset = offsets[i] / 10.0
        image = images[i]
        transformed[i] = intensity_shift(vertical_shift(horizontal_shift(image, offset[0]), offset[1]), offset[2])
    return transformed

def horizontal_shift(image, offset):
    x_dim = image.shape[1]
    offset = int(x_dim * offset)
    transformed = np.zeros(image.shape)
    for i in range(x_dim):
        transformed[:,i] = image[:,(i-offset) % x_dim]
    return transformed

def vertical_shift(image, offset):
    y_dim = image.shape[0]
    offset = int(y_dim * offset)
    transformed = np.zeros(image.shape)
    for i in range(y_dim):
        transformed[i] = image[(i-offset) % y_dim]
    return transformed

def intensity_shift(image, offset):
    transformed = np.clip(image + offset, 0, 1)
    return transformed

test()                                    
