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
import vis

l2 = 1e-5         # l2 weight decay

nc = 3            # # of channels in image
npx = 64
npy = 64          # # of pixels width/height of images
nx = npx*npy*nc   # # of dimensions in X

nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer

nbatch = 128      # # of examples in batch
nepoch = 10
ndisc = 1         # # of discr updates for each generator update

# Ported from https://github.com/Newmu/dcgan_code/blob/master/faces/train_uncond_dcgan.py
# 64x64
def generator_layers():
    layers = []
    wd = WeightRegularizer(l2=l2)
    assert npx % 16 == 0
    assert npy % 16 == 0
    layers.append(Dense(output_dim=ngf*8*(npx/16)*(npy/16), W_regularizer=wd))
    layers.append(BatchNormalization())
    layers.append(Activation('relu'))
    layers.append(Reshape((npx/16, npy/16, ngf*8)))
    deconv = False
    for wide  in (4, 2, 1):
        wd = WeightRegularizer(l2=l2)
        if deconv:
            layers.append(Deconvolution2D(ngf*wide, 5, 5,
                                          output_shape=(nbatch, npx/wide/2, npy/wide/2, ngf*wide),
                                          subsample=(2, 2), border_mode='same',
                                          W_regularizer=wd))
        else:
            layers.append(UpSampling2D(size=(2, 2)))
            layers.append(Convolution2D(ngf*wide, 5, 5, border_mode='same', W_regularizer=wd))
            layers.append(BatchNormalization())
            layers.append(Activation('relu'))

    if deconv:
        wd = WeightRegularizer(l2=l2)
        layers.append(Deconvolution2D(nc, 5, 5, output_shape=(nbatch, nc, npx, npx),
                                      subsample=(2, 2), border_mode='same', W_regularizer=wd))
    else:
        wd = WeightRegularizer(l2=l2)
        layers.append(UpSampling2D(size=(2, 2)))
        layers.append(Convolution2D(nc, 5, 5, border_mode='same', W_regularizer=wd))
        layers.append(Activation('tanh'))
    return layers


# 64x64
def discriminator_layers():
    alpha = 0.2
    layers=[]
    wd = WeightRegularizer(l2=l2)
    layers.append(Convolution2D(ndf, 5, 5, border_mode='same', W_regularizer=wd))
    layers.append(LeakyReLU(alpha=alpha))
    layers.append(BatchNormalization())
    layers.append(Activation('relu'))
    for wide in [2, 4, 8]:
        wd = WeightRegularizer(l2=l2)
        print "hey", ndf*wide
        layers.append(Convolution2D(ndf*wide, 5, 5,
                                    border_mode='same', subsample=(2, 2), W_regularizer=wd))
        layers.append(BatchNormalization())
        layers.append(LeakyReLU(alpha=alpha))
    layers.append(Flatten())
    wd = WeightRegularizer(l2=l2)
    layers.append(Dense(1, activation='sigmoid', W_regularizer=wd))
    return layers

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

############################################
disc_layers = discriminator_layers()
gen_layers = generator_layers()

gen_input = Input(shape=(nz,), name="gen_input")
disc_input = Input(shape=(npx, npy, nc), name="disc_input")

gen_output = gen_input
disc_output = disc_input
gen_disc_output = gen_output

for layer in gen_layers:
    gen_output = layer(gen_output)
    gen_disc_output = layer(gen_disc_output)
for layer in disc_layers:
    disc_output = layer(disc_output)
    gen_disc_output = layer(gen_disc_output)

discriminator = Model(disc_input, disc_output)
discriminator.compile(optimizer=Adam(), loss="mse")
print "Discriminator"
discriminator.summary()
#out = discriminator.predict([np.random.uniform(size=(nbatch, npx, npy, nc))])
#print out.shape
generator = Model(input=gen_input, output=gen_output)
generator.compile(optimizer=Adam(), loss="mse")
print "Generator:"
generator.summary()
#out = generator.predict([np.random.normal(size=(nbatch, nz))])
#print out.shape
gen_disc = Model(input=gen_input, output=gen_disc_output)

# combined net
#gen_disc = Sequential()
#gen_disc.add(generator)
#discriminator.trainable=False
#gen_disc.add(discriminator)
gen_disc.compile(loss='mse', optimizer=Adam())
print "combined net"
gen_disc.summary()

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
#make_trainable(discriminator, False)




print "loading data"
(x_train, x_test) = data.load("bedroom", trainSize=1000, testSize=500, shape=(64,64))

fake_gen_input = np.zeros(shape=(nbatch*2, nz)).astype('float32')
def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))

vis.plotImages(x_train[:100], 10, 10, "pictures/dcgan-orig")
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
        make_trainable(discriminator, True)
        disc_r = discriminator.fit(xs, ys, verbose=0, batch_size=nbatch, nb_epoch=1, shuffle=True)

        if batch % ndisc == ndisc-1: # update generator
            gen_in = np.random.normal(size=(nbatch, nz))
            gen_out = np.array([1.0] *  nbatch)
            make_trainable(discriminator, False)
            gen_r= gen_disc.fit(gen_in, gen_out, verbose=0, batch_size=nbatch, nb_epoch=1)
    
    disc_loss = disc_r.history['loss'][0]
    gen_loss = gen_r.history['loss'][0]

    print "Generator: {}, Discriminator: {}, Sum: {}".format(gen_loss, disc_loss, gen_loss+disc_loss)
    vis.displayRandom(10, x_train, nz, gaussian_sampler, generator, "pictures/dcgan-random", batch_size=nbatch)
