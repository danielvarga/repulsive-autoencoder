from keras.datasets import mnist
import os
import os.path
from PIL import Image
import numpy as np
import scipy.misc

def load(dataset, trainSize, testSize, shape=None, color=False, digit=None):
    if dataset == "mnist":
        (x_train, x_test) = load_mnist(digit)
    elif dataset == "celeba":
        # What's the pythonic way of doing this?
        (x_train, x_test) = load_celeba(shape=shape, color=color)  if shape is not None else load_celeba(color=color)
    elif dataset == "bedroom":
        (x_train, x_test) = load_bedroom(shape=shape, trainSize=trainSize, testSize=testSize)
    else:
        raise Exception("Invalid dataset: ", dataset)

    # if the feature dimension is missing, add it (!!! only works with Tensorflow !!!)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, 3)
        x_test = np.expand_dims(x_test, 3)

    if trainSize > 0:
        x_train = x_train[:trainSize]
    if testSize > 0:
        x_test = x_test[:testSize]

    return (x_train, x_test)

def load_mnist(digit=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    if digit is not None:
        mask_train = (y_train == digit)
        mask_test = (y_test == digit)
        x_train = x_train[mask_train]
        x_test = x_test[mask_test]
    return (x_train, x_test)

def load_celeba(shape=(72, 60),color=False):
    if shape==(72, 60):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-60x72"
        if color:
            cacheFile = "/home/zombori/datasets/celeba_color.npy"
        else:
            cacheFile = "/home/csadrian/datasets/celeba.npy"
    elif shape==(72, 64):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-64x72"
        if color:
            cacheFile = "/home/zombori/datasets/celeba6472_color.npy"
        else:
            cacheFile = "/home/csadrian/datasets/celeba6472.npy"
    else:
        assert False, "We don't have a celeba dataset with this size. Maybe you forgot about height x width order?"

    trainSize = 50000
    testSize = 5000
    if os.path.isfile(cacheFile):
        input = np.load(cacheFile)
    else:
        imgs = []
        height = None
        width = None
        for f in os.listdir(directory):
            if f.endswith(".jpg") or f.endswith(".png"):
                if color:
                    img = Image.open(os.path.join(directory, f))
                else:
                    img = Image.open(os.path.join(directory, f)).convert("L")                
                arr = np.array(img)
                if height is None:
                    height, width = arr.shape[:2]
                else:
                    assert (height, width) == arr.shape[:2], "Bad size %s %s" % (f, str(arr.shape))
                imgs.append(arr)
        input = np.array(imgs)
        input = input[:trainSize + testSize] / 255.0 # the whole dataset does not fit into memory as a float
        np.save(cacheFile,input)

    x_train = input[:trainSize]
    x_test = input[trainSize:trainSize+testSize]
    np.random.seed(10) # TODO Not the right place to do this.
    x_train = np.random.permutation(x_train)
    x_test = np.random.permutation(x_test)
    return (x_train, x_test)

def load_bedroom(shape=(64, 64), trainSize=0, testSize=0):
    if shape==(64, 64):
        cacheFile = "/home/zombori/datasets/bedroom/bedroom_64_64.npy"
    else:
        assert False, "We don't have a bedroom dataset with this size."
    if os.path.isfile(cacheFile):
        input = np.load(cacheFile)
    else:
        assert False, "Missing cache file: {}".format(cacheFile)
    if trainSize > 0:
        x_train = input[:trainSize]
    if testSize > 0:
        x_test = input[trainSize:trainSize+testSize]
    else:
        x_test = input[-200:] # TODO
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return (x_train, x_test)

        

def resize_bedroom(sizeX, sizeY, count, outputFile):
    directory = "/home/zombori/datasets/bedroom/data"
    def auxFun(path, count):
        if count <= 0: return (0, [])        
        if path.endswith('.webp'):
            img = Image.open(path)
            arr = np.array(img)
            arr = scipy.misc.imresize(arr, size=(sizeX, sizeY, 3))
            return (1, [arr])
        
        images=[]
        imgCount = 0
        for f in sorted(os.listdir(path)):
            f = os.path.join(path, f)
            currCount, currImages = auxFun(f, count - imgCount)
            images.extend(currImages)
            imgCount += currCount
        return (imgCount, images)
    cnt, images = auxFun(directory, count)
    images = np.array(images)
    np.save(outputFile, images)
    return images
