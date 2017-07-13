from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import cm
from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg16
from keras import backend as K
from keras.layers import Input

import vis
import load_models
import data

# set random seed
np.random.seed(10)

def image_data_format():
    return "channels_last"
K.image_data_format = image_data_format

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('--image_path', dest='image_path', default=None, type=str, help='Path to the image that we want to invert.')
parser.add_argument('--result_prefix', dest='result_prefix', type=str, help='Prefix for the saved results.')
parser.add_argument('--generator_prefix', dest='generator_prefix', type=str, help='Prefix for the saved generator.')

args = parser.parse_args()
image_path = args.image_path
result_prefix = args.result_prefix
generator_prefix = args.generator_prefix

# dimensions of the generated picture.
img_height = 64
img_width = 64
channels = 3
frequency = 10
iterations = 50
opt_iter = 50
tsne_points = 100

# some settings we found interesting
saved_settings = {
    'bad_trip': {'features': {'block4_conv1': 0.05,
                              'block4_conv2': 0.01,
                              'block4_conv3': 0.01},
                 'continuity': 0.1,
                 'dream_l2': 0.8,
                 'jitter': 5},
    'dreamy': {'features': {'block5_conv1': 0.05,
                            'block5_conv2': 0.02},
               'continuity': 0.1,
               'dream_l2': 0.02,
               'jitter': 0},
    'critic': {'features': {},
               'continuity': 0.0,
               'dream_l2': 0.0,
               'jitter': 0},
    'generator': {'features': {},
                  'continuity': 0.1,
                  'dream_l2': 0.0,
                  'jitter': 0},
}
# the settings we will use in this experiment
settings = saved_settings['generator']


def preprocess_image(image_path):
    # util function to open, resize and format pictures
    # into appropriate tensors
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img


def deprocess_image(x):
    # util function to convert a tensor into a valid image
    if K.image_data_format() == 'channels_first':
        x = x.reshape((channels, img_height, img_width))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_height, img_width, channels))
    x *= 255

    # # Remove zero-center by mean pixel
    # x[:, :, 0] += 103.939
    # x[:, :, 1] += 116.779
    # x[:, :, 2] += 123.68
    # # 'BGR'->'RGB'
    # x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if K.image_data_format() == 'channels_first':
    img_size = (channels, img_height, img_width)
else:
    img_size = (img_height, img_width, channels)

# load model
model = load_models.loadModel(generator_prefix)
print('Model loaded.')
model.summary()
latent_shape = K.int_shape(model.input)
batch_size = latent_shape[0]

# this will contain our inverted image
dream = Input(batch_shape=latent_shape)
generated = model(dream)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


def continuity_loss(x, singleImage):
    # continuity loss util function
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_height - 1, :img_width - 1] -
                     x[:, :, 1:, :img_width - 1])
        b = K.square(x[:, :, :img_height - 1, :img_width - 1] -
                     x[:, :, :img_height - 1, 1:])
    else:
        a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                     x[:, 1:, :img_width - 1, :])
        b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                     x[:, :img_height - 1, 1:, :])
        if singleImage:
            a = a[0]
            b = b[0]
    return K.sum(K.pow(a + b, 1.25))

if image_path is None:
    (x_train, target_image) = data.load("celeba", batch_size, batch_size, shape=(img_height, img_width), color=True)
    loss = K.sum(K.square(generated - target_image))
    singleImage = False
else:
    target_image = preprocess_image(image_path)
    loss = K.sum(K.square(generated[0] - target_image))
    singleImage = True


# add continuity loss (gives image local coherence, can result in an artful blur)
loss += settings['continuity'] * continuity_loss(generated, singleImage)

# compute the gradients of the dream wrt the loss
grads = K.gradients(loss, dream)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([dream], outputs)


def eval_loss_and_grads(x):
    x = x.reshape(latent_shape)
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    """Loss and gradients evaluator.
    This Evaluator class makes it possible
    to compute loss and gradients in one pass
    while retrieving them via two separate functions,
    "loss" and "grads". This is done because scipy.optimize
    requires separate functions for loss and gradients,
    but computing them separately would be inefficient.
    """

    def __init__(self):
        self.loss_value = None
        self.grad_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the loss
x = np.random.normal(size=latent_shape)
for i in range(iterations):
#    print('Start of iteration', i)
    start_time = time.clock()

    # Add a random jitter to the latent_code
    # This will be reverted at decoding time
    random_jitter = (settings['jitter'] * 2) * (np.random.random(latent_shape) - 0.5)
    x += random_jitter

    # Run L-BFGS for opt_iter steps
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=opt_iter)
    x = x.reshape(latent_shape)
    x -= random_jitter
    if (i+1) % frequency == 0:        
        # Decode the dream and save it
        img = model.predict(x, batch_size=batch_size)
        fname = result_prefix + '_inverted_%d' % (i+1)
        loss_value = min_val / np.prod(img_size)
        if image_path is None:
            imgs = vis.mergeSets((target_image, img))
            vis.plotImages(imgs, 2*10, batch_size//10, fname)
            loss_value /= batch_size 
        else:        
            img = np.expand_dims(img[0], axis=0)
            imgs = np.concatenate((target_image, img), axis=0)
            vis.plotImages(imgs, 2, 1, fname)
        print('Current loss value:', loss_value)
        end_time = time.clock()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i+1, end_time - start_time))

fig, ax = plt.subplots()

if ax is None:
    ax = plt.gca()


if True:
    from sklearn.manifold import TSNE
    import sklearn
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, metric="euclidean")
    reduced = tsne.fit_transform(x)

    target_image *= 255
    target_image = np.clip(target_image, 0, 255).astype('uint8')

    for i in range(tsne_points):
        x = reduced[i, 0]
        y = reduced[i, 1]
        im_a = target_image[i]
        image = Image.fromarray(im_a, mode="RGB")
        im = OffsetImage(image, zoom=0.5)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)

#    plt.figure(figsize=(12,12))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.savefig(result_prefix + "_inverted_tsne.png")
    plt.close()
