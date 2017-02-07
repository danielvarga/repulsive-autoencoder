import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import WeightRegularizer

import data

l2 = 1e-5         # l2 weight decay

nc = 3            # # of channels in image
npx = 64
npy = 64          # # of pixels width/height of images
nx = npx*npy*nc   # # of dimensions in X

nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer

nbatch = 128      # # of examples in batch
nepoch = 100
ndisc = 1         # # of discr updates for each generator update

# Ported from https://github.com/Newmu/dcgan_code/blob/master/faces/train_uncond_dcgan.py
# 64x64
def generator_layer():
    input = Input(shape=(nz,), name="gen_input")
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
    input = Input(shape=(npx, npy, nc), name="disc_input")
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


disc_input, disc_output = discriminator_layer()
discriminator = Model(input=disc_input, output=disc_output)
optimizer = Adam()
discriminator.compile(optimizer, loss="mse")
print "Discriminator"
discriminator.summary()
out = discriminator.predict([np.random.uniform(size=(nbatch, npx, npy, nc))])
print out.shape
gen_input, gen_output = generator_layer()
generator = Model(input=gen_input, output=gen_output)
optimizer = Adam()
generator.compile(optimizer, loss="mse")
print "Generator:"
generator.summary()
out = generator.predict([np.random.normal(size=(nbatch, nz))])
print out.shape

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
make_trainable(discriminator, False)


# combined net
gen_disc = Sequential()
gen_disc.add(generator)
discriminator.trainable=False
gen_disc.add(discriminator)
gen_disc.compile(loss='mse', optimizer=Adam())
print "combined net"
gen_disc.summary()


print "loading data"
(x_train, x_test) = data.load("bedroom", trainSize=1000, testSize=500, shape=(64,64))

fake_gen_input = np.zeros(shape=(nz,)).astype('float32')

print "starting training"
for epoch in range(nepoch):
    indices = np.random.permutation(range(x_train.shape[0]))
    for batch in range(x_train.shape[0] // nbatch):
        # update discriminator
        x_predicted = generator.predict([np.random.normal(size=(nbatch, nz))], batch_size=nbatch)
        curr_indices = indices[batch*nbatch: (batch+1)*nbatch]
        x_true = x_train[curr_indices]
        xs = np.concatenate((x_predicted, x_true), axis=0).astype('float32')
        ys = np.array([0] * nbatch + [1] * nbatch).astype('float32')
        d = {"disc_input":xs, "gen_input":fake_gen_input}
        r = discriminator.fit(d, ys, batch_size=nbatch, nb_epoch=1, shuffle=True)
        print 'E:',np.exp(r.totals['loss']/nbatch)

        if batch % ndisc == ndisc-1: # update generator
            gen_in = np.random.normal(size=(nbatch, nz))
            gen_out = np.array([1.0] *  nbatch)
            r= gen_disc.fit(inp, out, batch_size=nbatch, nb_epoch=1)
            print 'D:',np.exp(r.totals['loss']/nbatch)
    
