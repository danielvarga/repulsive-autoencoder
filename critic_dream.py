from __future__ import print_function

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
def image_data_format():
    return "channels_last"
K.image_data_format = image_data_format

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('--base_image_path', dest='base_image_path', type=str, help='Path to the image to transform.')
parser.add_argument('--result_prefix', dest='result_prefix', type=str, help='Prefix for the saved results.')
parser.add_argument('--discriminator_prefix', dest='discriminator_prefix', type=str, help='Prefix for the saved discriminator.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix
discriminator_prefix = args.discriminator_prefix

# dimensions of the generated picture.
img_height = 64
img_width = 64
channels = 3
frequency = 10
iterations = 50
opt_iter = 100

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

if K.image_data_format() == 'channels_first':
    img_size = (channels, img_height, img_width)
else:
    img_size = (img_height, img_width, channels)
# this will contain our generated image
dream = Input(batch_shape=(1,) + img_size)

# build the VGG16 network with our placeholder
# the model will be loaded with pre-trained ImageNet weights
model = vis.loadModel(discriminator_prefix)
print('Model loaded.')
model.summary()

#batch_size = K.int_shape(model.input)[0]
#dream_repeated = K.repeat_elements(dream, rep=batch_size, axis=0)
judgement = model(dream)

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


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
loss = - 10000 * judgement

# add continuity loss (gives image local coherence, can result in an artful blur)
loss += settings['continuity'] * continuity_loss(dream) / np.prod(img_size)
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
    x = x.reshape((1,) + img_size)
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
x = preprocess_image(base_image_path)
img = deprocess_image(np.copy(x))
fname = result_prefix + '_dream.png'
imsave(fname, img)
for i in range(iterations):
#    print('Start of iteration', i)
    start_time = time.clock()

    # Add a random jitter to the initial image.
    # This will be reverted at decoding time
#    random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
#    x += random_jitter

    # Run L-BFGS for opt_iter steps
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=opt_iter, pgtol=1e-9)
    print (info['task'])
    x = x.reshape(img_size)
#    x -= random_jitter
    if (i+1) % frequency == 0:        
        print('Current loss value:', min_val, ' grad norm: ', np.sum(np.square(info['grad'])))
        # Decode the dream and save it
        img = deprocess_image(np.copy(x))
        fname = result_prefix + '_dream_%d.png' % (i+1)
        imsave(fname, img)
        end_time = time.clock()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i+1, end_time - start_time))
