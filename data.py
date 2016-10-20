from keras.datasets import mnist
import os
import os.path
from PIL import Image
import numpy as np

def load(dataset):
    if dataset == "mnist":
        (x_train, x_test), (height, width) = load_mnist()
    elif dataset == "celeba":
        (x_train, x_test), (height, width) = load_celeba()
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

def load_celeba():
    directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-scale4"
    cacheFile = "/home/zombori/tmp/celeba.npy"
    trainSize = 10000
    testSize = 1000
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

    x_train = input[:trainSize]
    x_test = input[trainSize:trainSize+testSize]
    np.random.seed(10) # TODO Not the right place to do this.
    x_train = np.random.permutation(x_train)
    x_test = np.random.permutation(x_test)
    return (x_train, x_test), (height, width)
