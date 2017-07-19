from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
def image_data_format():
    return "channels_last"
K.image_data_format = image_data_format

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('--image_path', dest='image_path', default=None, type=str, help='Path to the image to transform.')
parser.add_argument('--prefix', dest='prefix', type=str, help='Prefix for the saved results and models.')

args = parser.parse_args()
image_path = args.image_path
prefix = args.prefix
discriminator_prefix = args.prefix + "_discriminator"
generator_prefix = args.prefix + "_generator"
gendisc_prefix = args.prefix + "_gendisc"

if image_path is not None:
    image_given = True
    batch_size = 1
else:
    generator = load_models.loadModel(generator_prefix)
    gendisc = load_models.loadModel(gendisc_prefix)
    batch_size = K.int_shape(generator.input)[0]
    latent_dim = K.int_shape(generator.input)[1]
    image_given = False


# dimensions of the generated picture.
img_height = 64
img_width = 64
channels = 3
frequency = 1
iterations = 10
opt_iter = 10

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
               'jitter': 0.0},
}
# the settings we will use in this experiment
settings = saved_settings['critic']


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
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def save_image(x, image_given, name):
    if image_given:
       img = deprocess_image(np.copy(x))
       imsave(name + ".png", img)
    else:
        y = generator.predict(x, batch_size = batch_size)
        vis.plotImages(y, 20, batch_size // 20, name)
        
if K.image_data_format() == 'channels_first':
    img_size = (channels, img_height, img_width)
else:
    img_size = (img_height, img_width, channels)


if image_given:
    model = load_models.loadModel(discriminator_prefix)
    input_batch_size = (1,) + img_size
    dream = Input(batch_shape=input_batch_size)
    judgement = model(dream)
else:
    model = load_models.loadModel(gendisc_prefix)
    input_batch_size = (batch_size, latent_dim)
    dream = Input(batch_shape=input_batch_size)
    judgement = K.mean(model(dream))


# get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers])


def continuity_loss(x):
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
    return K.sum(K.pow(a + b, 1.25))

# define the loss
loss = - judgement

# add continuity loss (gives image local coherence, can result in an artful blur)
# loss += settings['continuity'] * continuity_loss(dream) / np.prod(img_size)
# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
# loss += settings['dream_l2'] * K.sum(K.square(dream)) / np.prod(img_size)

# compute the gradients of the dream wrt the loss
grads = K.gradients(loss, dream)


outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([dream], outputs)


def eval_loss_and_grads(x):
    x = x.reshape(input_batch_size)
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
if image_given:
    x = preprocess_image(image_path)
else:
    x = np.random.normal(size=(batch_size, latent_dim))
    
fname = prefix + '_dream'
save_image(x, image_given, fname)
min_val = - np.mean(K.eval(model(x)))
print("Initial loss: ", min_val)

for i in range(iterations):
#    print('Start of iteration', i)
    start_time = time.clock()

    # Add a random jitter to the initial image.
    # This will be reverted at decoding time
#    random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
#    x += random_jitter

    # Run L-BFGS for opt_iter steps
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=opt_iter, pgtol=1e-9)

    x = x.reshape(input_batch_size)
#    x -= random_jitter
    if (i+1) % frequency == 0:        
        print('Current loss value:', min_val, ' grad norm: ', np.sum(np.square(info['grad'])))
        # Decode the dream and save it
        fname = prefix + '_dream_%d' % (i+1)
        save_image(x, image_given, fname)
        end_time = time.clock()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i+1, end_time - start_time))
