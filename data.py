from keras.datasets import mnist
import os
import os.path
from PIL import Image
import numpy as np
import scipy.misc
import scipy.ndimage

def load(dataset, trainSize, testSize, shape=None, color=False, digit=None):
    if dataset == "mnist":
        (x_train, x_test) = load_mnist(digit, shape)
    elif dataset == "celeba":
        # What's the pythonic way of doing this?
        (x_train, x_test) = load_celeba(shape=shape, color=color)  if shape is not None else load_celeba(color=color)
    elif dataset == "bedroom":
        (x_train, x_test) = load_bedroom(shape=(64,64), trainSize=trainSize, testSize=testSize)
    else:
        raise Exception("Invalid dataset: ", dataset)

    if trainSize > 0:
        x_train = x_train[:trainSize]
    if testSize > 0:
        x_test = x_test[:testSize]

    return (x_train, x_test)

def load_mnist(digit=None, shape=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # add_feature_dimension (!!! only works with Tensorflow !!!)
    x_train = np.expand_dims(x_train, 3)
    x_test = np.expand_dims(x_test, 3)

    if shape == (64, 64):
        print "!!!!!!!Loading upscaled mnist with 64*64 pixels!!!!!!!!!!"
        cacheFile = "/home/zombori/datasets/mnist_64_64.npz"
        if os.path.isfile(cacheFile):
            cache = np.load(cacheFile)
            x_train = cache["x_train"]
            x_test = cache["x_test"]
        else:
            x_train = resize_images(x_train, 64, 64, 1)
            x_test = resize_images(x_test, 64, 64, 1)
            np.savez(cacheFile, x_train=x_train, x_test=x_test)

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
    elif shape==(72, 64) or shape==(64,64):
        directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-64x72"
        if color:
            cacheFile = "/home/zombori/datasets/celeba6472_color.npy"
        else:
            cacheFile = "/home/csadrian/datasets/celeba6472.npy"
    else:
        assert False, "We don't have a celeba dataset with this size. Maybe you forgot about height x width order?"

    trainSize = 50000
    testSize = 4800 # It was 5000 for a long time, but Daniel wanted batch_size 400.
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

    if not color:
        input = np.expand_dims(input, 3)
    if shape==(64, 64):
        print "Truncated faces to get shape", shape
        input = input[:,4:68,:,:]

    x_train = input[:trainSize]
    x_test = input[trainSize:trainSize+testSize]
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
    print "Bedroom dataset size: ", input.shape
    if trainSize == 0: trainSize = 100000
    if testSize == 0: testSize = 200

    x_train = input[:trainSize]
    x_test = input[trainSize:trainSize+testSize]
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

def resize_images(dataset, sizeX, sizeY, sizeZ, outputFile=None):
    result = []
    for i in range(dataset.shape[0]):
        image = dataset[i]
        image_resized = scipy.ndimage.zoom(image, zoom=(1.0 * sizeX / image.shape[0], 1.0 * sizeY / image.shape[1], 1.0 * sizeZ / image.shape[2]))
        result.append(image_resized)
    result = np.array(result)
    if outputFile is not None: 
        np.save(outputFile, result)
    return result
                    
