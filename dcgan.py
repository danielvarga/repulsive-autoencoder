import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import WeightRegularizer


l2 = 1e-5         # l2 weight decay

nc = 3            # # of channels in image
npx = 64
npy = 64          # # of pixels width/height of images
nx = npx*npy*nc   # # of dimensions in X

nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer

nbatch = 128      # # of examples in batch


# Ported from https://github.com/Newmu/dcgan_code/blob/master/faces/train_uncond_dcgan.py
# 64x64
def generator_layer():
    input = Input(shape=(nz,))
    net = input
    wd = WeightRegularizer(l2=l2)
    assert npx % 16 == 0
    assert npy % 16 == 0
    net = Dense(output_dim=ngf*8*(npx/16)*(npy/16), W_regularizer=wd)(net)
    net = Activation('relu')(BatchNormalization()(net))
    net = Reshape((npx/16, npy/16, ngf*8))(net)
    deconv = False
    for wide  in (4, 2, 1):
        wd = WeightRegularizer(l2=l2)
        if deconv:
            net = Deconvolution2D(ngf*wide, 5, 5,
                output_shape=(nbatch, npx/wide/2, npy/wide/2, ngf*wide),
                subsample=(2, 2), border_mode='same',
                W_regularizer=wd)(net)
        else:
            net = UpSampling2D(size=(2, 2))(net)
            net = Convolution2D(ngf*wide, 5, 5, border_mode='same', W_regularizer=wd)(net)
        net = Activation('relu')(BatchNormalization()(net))

        if deconv:
            wd = WeightRegularizer(l2=l2)
            net = Deconvolution2D(nc, 5, 5, output_shape=(nbatch, nc, npx, npx),
                                  subsample=(2, 2), border_mode='same', W_regularizer=wd)(net)
        else:
            wd = WeightRegularizer(l2=l2)
            net = UpSampling2D(size=(2, 2))(net)
            net = Convolution2D(nc, 5, 5, border_mode='same', W_regularizer=wd)(net)
    net = Activation('tanh')(net)
    return input, net


# 64x64
def discriminator_layer():
    alpha = 0.2
    input = Input(shape=(npx, npy, nc))
    net = input
    wd = WeightRegularizer(l2=l2)
    net = Convolution2D(ndf, 5, 5, border_mode='same', W_regularizer=wd)(net)
    net = LeakyReLU(alpha=alpha)(net)
    net = Activation('relu')(BatchNormalization()(net))
    for wide in [2, 4, 8]:
        wd = WeightRegularizer(l2=l2)
        print "hey", ndf*wide
        net = Convolution2D(ndf*wide, 5, 5,
            border_mode='same', subsample=(2, 2), W_regularizer=wd)(net)
        net = LeakyReLU(alpha=alpha)(BatchNormalization()(net))
    net = Flatten()(net)
    wd = WeightRegularizer(l2=l2)
    net = Dense(1, activation='sigmoid', W_regularizer=wd)(net)
    return input, net


# From https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
def generator_layer_mnist():
    input = Input(shape=(100,))
    net = input
    net = Dense(input_dim=100, output_dim=1024)(net)
    net = Activation('tanh')(net)
    net = Dense(128*7*7)(net)
    net = BatchNormalization()(net)
    net = Activation('tanh')(net)
    net = Reshape((128, 7, 7), input_shape=(128*7*7,))(net)
    net = UpSampling2D(size=(2, 2))(net)
    net = Convolution2D(64, 5, 5, border_mode='same')(net)
    net = Activation('tanh')(net)
    net = UpSampling2D(size=(2, 2))(net)
    net = Convolution2D(1, 5, 5, border_mode='same')(net)
    net = Activation('tanh')(net)
    return input, net


# https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
def discriminator_layer_mnist():
    input = Input(shape=(1, 28, 28))
    net = input
    net = Convolution2D(64, 5, 5,
                        border_mode='same',
                        input_shape=(1, 28, 28))(net)
    net = Activation('tanh')(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Convolution2D(128, 5, 5)(net)
    net = Activation('tanh')(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Flatten()(net)
    net = Dense(1024)(net)
    net = Activation('tanh')(net)
    net = Dense(1)(net)
    net = Activation('sigmoid')(net)
    return input, net


input, output = discriminator_layer()
model = Model(input=input, output=output)
optimizer = Adam()
model.compile(optimizer, loss="mse")
model.summary()
out = model.predict([np.random.uniform(size=(nbatch, npx, npy, nc))])
print out.shape
input, output = generator_layer()
model = Model(input=input, output=output)
optimizer = Adam()
model.compile(optimizer, loss="mse")
model.summary()
out = model.predict([np.random.normal(size=(nbatch, nz))])
print out.shape
