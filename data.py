from keras.datasets import mnist
import os
import os.path
import random
from PIL import Image
import numpy as np
import scipy.misc
import scipy.ndimage
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import annoy

import vis

if K.image_dim_ordering() == 'th':
    feature_axis = 1
elif K.image_dim_ordering() == 'tf':
    feature_axis = 3
else:
    assert False, "Unknown dim ordering"


# returns an object with
# properties
#    name
#    shape
#    color
#    finite
#    synthetic
# guaranteed methods
#    get_data(trainSize, testSize): -> (x_train, x_test)
#    get_train_flow(batch_size): -> object with next() method to give batch_size number of samples
#    get_nearest_samples(generated_samples)
# guaranteed methods for finite subclasses
#    get_finite_set(): -> x_train
# guaranteed methods for infinte synthetic subclasses
#    get_uniform_data():

def load(dataset, shape=None, color=True):
    if dataset == "mnist":
        return Dataset_mnist(shape)
    elif dataset == "celeba":
        return Dataset_celeba(shape, color)
    elif dataset == "bedroom":
        return Dataset_bedroom(shape)    
    elif dataset == "syn-circles":
        return Dataset_circles_centered(shape)
    elif dataset == "syn-moving-circles":
        return Dataset_moving_circles(shape)
    elif dataset == "syn-rectangles":
        return Dataset_syn_rectangles(shape)
    elif dataset == "syn-gradient":
        return Dataset_syn_gradient(shape)
    elif dataset == "syn-constant-uniform":
        return Dataset_syn_constant_uniform(shape)
    else:
        raise Exception("Invalid dataset: ", dataset)

def test(file):
    datasets ["mnist", "celeba", "bedroom", "syn-circles", "syn-moving-circles", "syn-rectangles", "syn-gradient"]
    shape=(64, 64)
    trainSize = 20
    testSize = 1
    color = True
    result = []
    for dataset in datasets:
        print "Testing dataset: {}".format(dataset)
        data_object = load(dataset, shape, color)
        x_train, x_test = data_object.get_data(trainSize, testSize)
        if x_train.shape[feature_axis] == 1:
            x_train = np.concatenate([x_train, x_train, x_train], axis=feature_axis)
        result.append(x_train)
        x_batch = data_object.get_train_flow(trainSize).next()
        if x_batch.shape[feature_axis] == 1:
            x_batch = np.concatenate([x_batch, x_batch, x_batch], axis=feature_axis)
        result.append(x_batch)

    result = np.concatenate(result)
    vis.plotImages(result, trainSize, 2*len(datasets), file)


def test_uniform(file):
    datasets = ["syn-rectangles", "syn-gradient"]
    shape = (64,64)
    color = False
    result = []
    for dataset in datasets:
        print "Testing dataset: {}".format(dataset)
        data_object = load(dataset, shape, color)
        x_uniform = data_object.get_uniform_data()
        result.append(x_uniform[:400])

    result = np.concatenate(result)
    vis.plotImages(result, 20, 20*len(datasets), file)


class Dataset(object):
    def __init__(self, name, shape, color=False, finite=False, synthetic=False):
        assert len(shape)==2, "Expected shape of length 2"
        self.name = name
        self.shape = shape
        self.color = color
        self.finite = finite
        self.synthetic = synthetic
    def get_data(self, trainSize, testSize):
        assert False, "Not Yet Implemented"
    def get_train_flow(self, batch_size):
        assert False, "Not Yet Implemented"
    def get_nearest_samples(self, generated_samples):
        trainSize = 1000
        x_train, x_test = self.get_data(trainSize, 1)
        x_true = x_train.reshape(trainSize, -1)
        x_generated = generated_samples.reshape(generated_samples.shape[0], -1)
        
        f = x_true.shape[1]
        t = annoy.AnnoyIndex(f, metric="euclidean")
        for i, v in enumerate(x_true):
            t.add_item(i, v)
        t.build(100)

        hist = np.zeros(len(x_true))
        result = []
        for g in x_generated:
            nearest_index = t.get_nns_by_vector(g, 1)[0]
            result.append(x_generated[nearest_index])
            hist[nearest_index] += 1
        result = np.array(result)
        return result, hist


class Dataset_real(Dataset):
    def __init__(self, name, shape, color=False):
        super(Dataset_real, self).__init__(name, shape, color=color, finite=False, synthetic=False)
    def get_train_flow(self, batch_size):
        imageGenerator = ImageDataGenerator()
        try:
            flow_object = imageGenerator.flow(self.x_train, batch_size = batch_size)
        except AttributeError:
            assert False, "You need to call get_data to instantiate self.x_train"
        return flow_object
    def get_normalized_image_data(self, input, trainSize, testSize):
        assert trainSize > 0 and testSize > 0, "trainSize and testSize must be positive"            
        x_train = input[:trainSize]
        x_test = input[trainSize:trainSize+testSize]
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        return(x_train, x_test)
    def limit_data(self, input, size):
        if size > 0:
            input = input[:size]
        return input

class Dataset_mnist(Dataset_real):
    def __init__(self, shape=(28,28)):
        super(Dataset_mnist, self).__init__("mnist", shape, color=False)

        cacheFile_64_64 = "/home/zombori/datasets/mnist_64_64.npz"
        if shape == (64, 64) and os.path.isfile(cacheFile_64_64):
            cache = np.load(cacheFile_64_64)
            self.x_train_orig = cache["x_train"]
            self.x_test_orig = cache["x_test"]
            return

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        # add_feature_dimension
        x_train = np.expand_dims(x_train, feature_axis)
        x_test = np.expand_dims(x_test, feature_axis)

        if shape == (64, 64):
            x_train = resize_images(x_train, 64, 64, 1)
            x_test = resize_images(x_test, 64, 64, 1)
            np.savez(cacheFile_64_64, x_train=x_train, x_test=x_test)        
        self.x_train_orig = x_train
        self.x_test_orig = x_test
    def get_data(self, trainSize, testSize):
        self.x_train = self.limit_data(self.x_train_orig, trainSize)
        self.x_test = self.limit_data(self.x_test_orig, testSize)
        return (self.x_train, self.x_test)

class Dataset_celeba(Dataset_real):
    def __init__(self, shape=(64,64), color=True):
        super(Dataset_celeba, self).__init__("celeba", shape, color)

        # determine cache file
        if shape==(72, 60):
            directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-60x72"
            if color:
                cacheFile = "/home/zombori/datasets/celeba_72_60_color.npy"
            else:
                cacheFile = "/home/zombori/datasets/celeba_72_60.npy"
        elif shape==(72, 64) or shape==(64,64):
            directory = "/home/daniel/autoencoding_beyond_pixels/datasets/celeba/img_align_celeba-64x72"
            if color:
                cacheFile = "/home/zombori/datasets/celeba_72_64_color.npy"
            else:
                cacheFile = "/home/zombori/datasets/celeba_72_64.npy"
        else:
            assert False, "We don't have a celeba dataset with this size. Maybe you forgot about height x width order?"

        # load input
        if os.path.isfile(cacheFile):
            self.input = np.load(cacheFile)
        else:
            imgs = []
            height = None
            width = None
            for f in sorted(os.listdir(directory)):
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
            self.input = np.array(imgs)
            np.save(cacheFile,self.input)

        if not color:
            self.input = np.expand_dims(self.input, feature_axis)
        if shape==(64, 64):
            print "Truncated faces to get shape", shape
            self.input = self.input[:,4:68,:,:]
    def get_data(self, trainSize, testSize):
        self.x_train, self.x_test = self.get_normalized_image_data(self.input, trainSize, testSize)
        return (self.x_train, self.x_test)

class Dataset_bedroom(Dataset_real):
    def __init__(self, shape=(64,64)):
        super(Dataset_bedroom, self).__init__("bedroom", shape, color=True)

        if shape==(64, 64):
            cacheFile = "/home/zombori/datasets/bedroom/bedroom_64_64.npy"
        else:
            assert False, "We don't have a bedroom dataset with size {}".format(shape)
        if os.path.isfile(cacheFile):
            self.input = np.load(cacheFile)
        else:
            assert False, "Missing cache file: {}".format(cacheFile)        
    def get_data(self, trainSize, testSize):
        self.x_train, self.x_test = self.get_normalized_image_data(self.input, trainSize, testSize)
        return (self.x_train, self.x_test)

class Dataset_synthetic(Dataset):
    def __init__(self, name, shape, finite):
        assert shape is not None, "Synthetic datasets must have a valid shape argument"
        super(Dataset_synthetic, self).__init__(name, shape=shape, color=False, finite=finite, synthetic=True)

class Dataset_syn_finite(Dataset_synthetic):
    def __init__(self, name, shape):
        super(Dataset_syn_finite, self).__init__(name, shape=shape, finite=True)
        self.generate_finite_set()
    def get_data(self, trainSize, testSize):
        assert trainSize > 0 and testSize > 0, "trainSize and testSize must be positive"            
        train_indices = np.random.choice(len(self.finite_set), trainSize)
        test_indices = np.random.choice(len(self.finite_set), testSize)
        self.x_train = self.finite_set[train_indices]
        self.x_test = self.finite_set[test_indices]
        return (self.x_train, self.x_test)
    def get_finite_set(self):
        return self.finite_set
    def get_train_flow(self, batch_size):
        class FiniteGenerator(object):
            def __init__(self, finite_set, batch_size):
                self.finite_set = finite_set
                self.batch_size = batch_size
                self.index_range = range(len(self.finite_set))
            def next(self):
                selected_indices = np.random.choice(self.index_range, self.batch_size)
                return self.finite_set[selected_indices]
        return FiniteGenerator(self.finite_set, batch_size)
    def get_nearest_samples(self, generated_samples):
        x_true = self.finite_set(self.finite_set.shape[0], -1)
        x_generated = generated_samples.reshape(generated_samples.shape[0], -1)
        
        f = x_true.shape[1]
        t = annoy.AnnoyIndex(f, metric="euclidean")
        for i, v in enumerate(x_true):
            t.add_item(i, v)
        t.build(100)

        hist = np.zeros(len(x_true))
        result = []
        for g in x_generated:
            nearest_index = t.get_nns_by_vector(g, 1)[0]
            result.append(x_generated[nearest_index])
            hist[nearest_index] += 1
        result = np.array(result)
        return result, hist
    def generate_finite_set(self): # TO BE OVERWRITTEN
        self.finite_set = None

class Dataset_circles_centered(Dataset_syn_finite):
    def __init__(self, shape):
        super(Dataset_circles_centered, self).__init__("syn-circles", shape=shape)
    def generate_finite_set(self):
        shape = self.shape
        max_radius = min(shape) // 2
        radius_range = range(max_radius + 1)
        
        data = np.zeros((max_radius + 1, shape[0], shape[1]))
        for r in radius_range:
            for y in range(shape[0]):
                for x in range(shape[1]):
                    if (x-max_radius)**2 + (y-max_radius)**2 < r**2:
                        data[r, y, x] = 1
        data = np.expand_dims(data, feature_axis)
        self.finite_set = data

class Dataset_moving_circles(Dataset_syn_finite):
    def __init__(self, shape):
        super(Dataset_moving_circles, self).__init__("syn-moving-circles", shape=shape)
    def generate_finite_set(self):
        shape = self.shape
        radius = min(shape) // 8
        y_range = range(radius, shape[0] - radius)
        x_range = range(radius, shape[1] - radius)
        set_size = len(y_range) * len(x_range)

        data = np.zeros((set_size, shape[0], shape[1]))
        for i in range(set_size):
            center_y = y_range[i // len(y_range)]
            center_x = x_range[i % len(y_range)]
            for y in range(shape[0]):
                for x in range(shape[1]):
                    if (x-center_x)**2 + (y-center_y)**2 < radius**2:
                        data[i, y, x] = 1
        data = np.expand_dims(data, feature_axis)
        self.finite_set = data

class Dataset_syn_infinite(Dataset_synthetic):
    def __init__(self, name, shape):
        super(Dataset_syn_infinite, self).__init__(name, shape=shape, finite=False)
    def get_data(self, trainSize, testSize):
        x_train = self.generate_samples(trainSize)
        x_test = self.generate_samples(testSize)
        return (x_train, x_test)
    def get_train_flow(self, batch_size):
        class Generator(object):
            def __init__(self, batch_size, generator):
                self.generator = generator
                self.batch_size = batch_size
            def next(self):
                return self.generator(batch_size)
        return Generator(batch_size, self.generate_samples)
    def get_uniform_data(self):
        samples = self.get_uniform_samples()
        data = np.zeros((len(samples), self.shape[0], self.shape[1]))
        for i, sample in enumerate(samples):
            self.generate_one_sample(data[i], sample)
        data = np.expand_dims(data, feature_axis)
        return data        
    def generate_samples(self, size):
        data = np.zeros((size, self.shape[0], self.shape[1]))
        for i in range(len(data)):
            self.generate_one_sample(data[i], self.sampler())
        data = np.expand_dims(data, feature_axis)
        return data
    def generate_one_sample(self, data, random_sample):
        assert False, "NYI"
    def sampler(self):
        assert False, "NYI"
    def get_uniform_samples(self):
        assert False, "NYI"

class Dataset_syn_rectangles(Dataset_syn_infinite):
    def __init__(self, shape):
        super(Dataset_syn_rectangles, self).__init__("syn-rectangles", shape=shape)
    def generate_one_sample(self, data, coordinates):
        assert len(coordinates) == 4
        h, w = data.shape
        ys = coordinates[:2] * (h+1)
        xs = coordinates[2:] * (w+1)
        ys = sorted(ys.astype(int))
        xs = sorted(xs.astype(int))
        data[ys[0]:ys[1], xs[0]:xs[1]] = 1
    def sampler(self):
        return np.random.uniform(size=4)
    def get_uniform_samples(self):
        size = 6
        samples = []
        for y1 in range(size-1):
            for y2 in range(y1+1, size):
                for x1 in range(size-1):
                    for x2 in range(x1+1, size):
                        sample = np.array([y1,y2,x1,x2]) * 1.0 / size 
                        samples.append(sample)
        samples = np.array(samples)
        return samples
        
class Dataset_syn_gradient(Dataset_syn_infinite):
    def __init__(self, shape):
        super(Dataset_syn_gradient, self).__init__("syn-gradient", shape=shape)
    def generate_one_sample(self, data, direction):
        h, w = data.shape
        assert h==w
        c, s = np.cos(direction), np.sin(direction)
        for y in range(h):
            for x in range(w):
                yy = 2 * float(y) / h - 1
                xx = 2 * float(x) / w - 1
                scalar_product = yy * s + xx * c
                normed = (scalar_product / np.sqrt(2) + 1) / 2 # even the 45 degree gradients are in [0, 1].
                data[y, x] = normed
    def sampler(self):
        return np.random.uniform(0.0, 2*np.pi)
    def get_uniform_samples(self):
        return np.linspace(0, 2*np.pi, 360, endpoint=False)

class Dataset_syn_constant_uniform(Dataset_syn_infinite):
    def __init__(self, shape):
        super(Dataset_syn_gradient, self).__init__("syn-constant-uniform", shape=shape)
    def generate_one_sample(self, data, level):
        data[:, :] = level
    def sampler(self):
        return np.random.uniform(0, 1)
    def get_uniform_samples(self):
        return np.linspace(0, 1, 1001, endpoint=True)
    def get_nearest_params(data):
        # to clip or not to clip.
        return data.mean(axis=(1, 2))


############################################################

"""
def load(dataset, trainSize, testSize, shape=None, color=False, digit=None):
    if dataset == "mnist":
        (x_train, x_test) = load_mnist(digit, shape)
    elif dataset == "celeba":
        # What's the pythonic way of doing this?
        (x_train, x_test) = load_celeba(shape=shape, color=color)  if shape is not None else load_celeba(color=color)
    elif dataset == "bedroom":
        (x_train, x_test) = load_bedroom(shape=(64,64), trainSize=trainSize, testSize=testSize)
    elif dataset == "syn-circles":
        x_train = generate_circles(shape=(64,64), size=trainSize)
        x_test  = generate_circles(shape=(64,64), size=testSize)
    elif dataset == "syn-moving-circles":
        x_train = generate_moving_circles(shape=(64,64), size=trainSize)
        x_test  = generate_moving_circles(shape=(64,64), size=testSize)
    elif dataset == "syn-moving-changing-circles":
        x_train = generate_several_circles(shape=(64,64), size=trainSize, circleCount=1)
        x_test  = generate_several_circles(shape=(64,64), size=testSize, circleCount=1)        
    elif dataset == "syn-circle-pairs":
        x_train = generate_several_circles(shape=(64,64), size=trainSize, circleCount=2)
        x_test  = generate_several_circles(shape=(64,64), size=testSize, circleCount=2)        
    elif dataset == "syn-rectangles":
        x_train = generate_rectangles(shape=(64,64), size=trainSize)
        x_test  = generate_rectangles(shape=(64,64), size=testSize)
    elif dataset == "syn-gradient":
        x_train = generate_general((64, 64), trainSize, single_gradient, single_gradient_sampler)
        x_test  = generate_general((64, 64), testSize,  single_gradient, single_gradient_sampler)
    else:
        raise Exception("Invalid dataset: ", dataset)

    if trainSize > 0:
        x_train = x_train[:trainSize]
    if testSize > 0:
        x_test = x_test[:testSize]

    return (x_train, x_test)


def generate_general(shape, size, generator, sampler):
    h, w = shape
    assert h==w
    data = np.zeros((size, h, w))
    for i in range(size):
        generator(data[i], sampler())
    return np.expand_dims(data, 3)


def generate_circles(shape, size):
    if size == 0: size = 1000
    assert len(shape)==2
    max_radius = min(shape) // 2
    data = np.zeros((size, shape[0], shape[1]))
    for i in range(size):
        r = random.randrange(max_radius + 1) # yeah, randint, screw randint :)
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (x-max_radius)**2 + (y-max_radius)**2 < r**2:
                    data[i, y, x] = 1
    return np.expand_dims(data, 3)

# fixed radius circle but at moving location
def generate_moving_circles(shape, size):
    if size == 0: size = 1000
    assert len(shape)==2
    radius = min(shape) // 4
    print size, shape, radius
    data = np.zeros((size, shape[0], shape[1]))
    for i in range(size):
        center_x = random.randrange(radius, shape[1] - radius + 1) # yeah, randint, screw randint :)
        center_y = random.randrange(radius, shape[0] - radius + 1) # yeah, randint, screw randint :)
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (x-center_x)**2 + (y-center_y)**2 < radius**2:
                    data[i, y, x] = 1
    return np.expand_dims(data, 3)
    

def generate_several_circles(shape, size, circleCount):
    if size == 0: size = 1000
    assert len(shape)==2
    data = np.zeros((size, shape[0], shape[1]))
    for i in range(size):
        # first generate center points
        centers = np.random.uniform(size=(circleCount,2)) * shape
        # determine maximum radii such that each circle fits in the picture
        distances_from_edge = np.concatenate([centers, shape-centers], axis=1)
        max_radii = np.min(distances_from_edge, axis=1)
        radii = np.random.uniform(size=circleCount, high=max_radii)

        for y in range(shape[0]):
            for x in range(shape[1]):
                for j in range(circleCount):
#                    r = radii[j]
                    r = 5.0
                    center_x = centers[j,0]
                    center_y = centers[j,1]
                    if (x-center_x)**2 + (y-center_y)**2 < r**2:
                        data[i, y, x] = 1
                        break
    return np.expand_dims(data, 3)

def generate_rectangles(shape, size):
    assert len(shape)==2
    data = np.zeros((size, shape[0], shape[1]))
    for i in range(size):
        ys = np.random.randint(shape[0]+1, size=2)
        ys = sorted(ys)
        xs = np.random.randint(shape[1]+1, size=2)
        xs = sorted(xs)
        data[i, ys[0]:ys[1], xs[0]:xs[1]] = 1
    return np.expand_dims(data, 3)

def load_mnist(digit=None, shape=None):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    np.savez("mnist-labels.npz", **{"train": y_train, "test": y_test})

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


def load_celeba(shape=(72, 60), color=False):
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
"""
        

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
                    
