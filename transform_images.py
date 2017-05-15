import numpy as np

import data
import vis

transformation_types = 3

def test():
    data_object = data.load("mnist", shape=(28, 28))
    x_train, x_test = data_object.get_data(2, 1)
    images = []
    for i in range(5):
        offsets = np.random.normal(size=(transformation_types))
        images.append(shift(x_train, offsets))
    # for offset in (0.2, 0.4, 0.6, 0.8):
    #     images.append(horizontal_shift(x_train, offset))
    #     images.append(vertical_shift(x_train, offset))
    # for offset in (-1, -0.5, 0.5, 1, 3):
    #     images.append(intensity_shift(x_train, offset))
    images = np.concatenate(images)
    vis.plotImages(images, 10, 4, "pictures/transform_images_test")

def shift(images, offsets):
    assert len(offsets) == transformation_types
    transformed = images
    transformed = horizontal_shift(transformed, offsets[0])
    transformed = vertical_shift(transformed, offsets[1])
    transformed = intensity_shift(transformed, offsets[2])
    return transformed

def horizontal_shift(images, offset):
    x_dim = images.shape[2]
    offset = int(x_dim * offset / 10.0)
    transformed = np.zeros(images.shape)
    for i in range(x_dim):
        transformed[:,:,i] = images[:,:,(i-offset) % x_dim]
    return transformed

def vertical_shift(images, offset):
    y_dim = images.shape[1]
    offset = int(y_dim * offset / 10.0)
    transformed = np.zeros(images.shape)
    for i in range(y_dim):
        transformed[:,i] = images[:,(i-offset) % y_dim]
    return transformed

def intensity_shift(images, offset):
    transformed = np.clip(images + offset / 4.0, 0, 1)
    return transformed

test()                                    
