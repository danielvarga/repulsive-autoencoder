from keras.datasets import mnist
import os
import os.path
from PIL import Image
import numpy as np

def load(dataset, shape=None):
    if dataset == "mnist":
        assert shape is None
        (x_train, x_test), (height, width) = load_mnist()
    elif dataset == "celeba":
        (x_train, x_test), (height, width) = load_celeba(shape=shape)
    else:
        raise Exception("Invalid dataset: ", dataset)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return (x_train, x_test), (height, width)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return (x_train, x_test), (28, 28)

def load_celeba(shape=(72, 60)):
    if shape==(72, 60):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-60x72"
        cacheFile = "/home/csadrian/datasets/celeba.npy"
    elif shape==(72, 64):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-64x72"
        cacheFile = "/home/csadrian/datasets/celeba6472.npy"
    else:
        assert False, "We don't have a celeba dataset with this size. Maybe you forgot about height x width order?"

    trainSize = 100000
    testSize = 10000
    if os.path.isfile(cacheFile):
        input = np.load(cacheFile)
        height, width = input.shape[1:]
    else:
        imgs = []
        height = None
        width = None
        for f in os.listdir(directory):
            if f.endswith(".jpg") or f.endswith(".png"):
                img = Image.open(os.path.join(directory, f)).convert("L")
                arr = np.array(img)
                if height is None:
                    height, width = arr.shape
                else:
                    assert (height, width) == arr.shape, "Bad size %s %s" % (f, str(arr.shape))
                imgs.append(arr)
        input = np.array(imgs).astype(float) / 255
        np.save(cacheFile,input)

    assert shape == (height, width), "Loaded dataset not compatible with prescribed shape."

    x_train = input[:trainSize]
    x_test = input[trainSize:trainSize+testSize]
    np.random.seed(10) # TODO Not the right place to do this.
    x_train = np.random.permutation(x_train)
    x_test = np.random.permutation(x_test)
    return (x_train, x_test), (height, width)
