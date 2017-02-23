import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import WeightRegularizer
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils

import keras.backend as K
import tensorflow as tf
import time

import dcgan_params
import data
import vis
import model_dcgan

args = dcgan_params.getArgs()
print(args)

# set random seed
np.random.seed(10)


# limit memory usage
import keras
print "Keras version: ", keras.__version__
if keras.backend._BACKEND == "tensorflow":
    import tensorflow as tf
    print "Tensorflow version: ", tf.__version__
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))



############################################
print "loading data"
(x_train, x_test) = data.load(args.dataset, trainSize=args.trainSize, testSize=args.testSize, shape=args.shape, color=args.color)
imageGenerator = ImageDataGenerator()
x_true_flow = imageGenerator.flow(x_train, batch_size = args.batch_size)


print "building networks"
disc_layers = model_dcgan.discriminator_layers_wgan(latent_dim=args.latent_dim, wd=args.wd, bn_allowed=args.use_bn_disc)
gen_layers = model_dcgan.generator_layers_wgan(latent_dim=args.latent_dim, batch_size=args.batch_size, wd=args.wd, bn_allowed=args.use_bn_gen, image_channel=x_train.shape[3])

gen_input = Input(batch_shape=(args.batch_size,args.latent_dim), name="gen_input")
disc_input = Input(batch_shape=(args.batch_size, args.shape[0], args.shape[1], x_train.shape[3]), name="disc_input")

gen_output = gen_input
disc_output = disc_input
gen_disc_output = gen_output

for layer in gen_layers:
    gen_output = layer(gen_output)
    gen_disc_output = layer(gen_disc_output)

for layer in disc_layers:
    disc_output = layer(disc_output)
    gen_disc_output = layer(gen_disc_output)

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    x = y_true * y_pred
    zeros = K.zeros(y_pred.shape)
    x_correct = K.maximum(x, zeros)
    x_false = K.minimum(x, zeros)
    loss = - args.falseWeight * K.sum(x_false) - K.sum(x_correct)
    return loss

def D_acc(y_true, y_pred):
    x = y_true * y_pred
    return 100 * K.mean(x > 0)
def D_acc_np(y_true, y_pred):
    x = y_true * y_pred
    return 100 * np.mean(x > 0)



discriminator = Model(disc_input, disc_output)
if args.optimizer == "adam":
    optimizer_d = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_d = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    optimizer_d = SGD(lr=args.lr)
discriminator.compile(optimizer=optimizer_d, loss=D_loss, metrics=[D_acc])
print "Discriminator"
discriminator.summary()

generator = Model(input=gen_input, output=gen_output)
gen_disc = Model(input=gen_input, output=gen_disc_output)
if args.optimizer == "adam":
    optimizer_g = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_g = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    optimizer_g = SGD(lr=args.lr)
generator.compile(optimizer=optimizer_g, loss="mse")
print "Generator:"
generator.summary()
gen_disc.compile(loss=D_loss, optimizer=optimizer_g)
print "combined net"
gen_disc.summary()

def ndisc(gen_iters):
    if gen_iters < 25:
        return 100
    elif gen_iters % 500 == 0:
	return 100
    else:
        return 5


def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))
vis.plotImages(x_train[:100], 10, 10, args.prefix + "-orig")

from keras.callbacks import Callback

class ClipperCallback(Callback):
    def __init__(self, layers, clipValue):
	self.layers = layers
        self.clipValue = clipValue

    def on_batch_begin(self):
        self.clip()

    def clip(self):
	for layer in self.layers:
#            if layer.__class__.__name__ not in ("Convolution2D", "BatchNormalization"): continue
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = np.clip(weights[i], - self.clipValue, self.clipValue)
            layer.set_weights(weights)
		    
if args.clipValue > 0:
    clipper = ClipperCallback(disc_layers, args.clipValue)
    callbacks = [clipper]
else:
    callbacks = []

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

y_generated = np.array([-1.0] *  args.batch_size).reshape((-1,1)).astype("float32")
y_true = np.array([1.0] *  args.batch_size).reshape((-1,1)).astype("float32")
ys = np.concatenate((y_generated, y_true), axis=0)

print "starting training"
startTime = time.clock()
for iter in range(args.nb_iter):
    # update discriminator
    discriminator.trainable = True
    for disc_iter in range(ndisc(iter)):
        x_true = x_true_flow.next()
        gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
        x_generated = generator.predict([gen_in], batch_size=args.batch_size)
        clipper.clip()
        disc_loss1 = discriminator.train_on_batch(x_true, y_true)
        disc_loss2 = discriminator.train_on_batch(x_generated, y_generated)
        disc_loss = disc_loss1[0] + disc_loss2[0]

    xs = np.concatenate((x_generated, x_true), axis=0)
    disc_pred = discriminator.predict(xs, batch_size=args.batch_size)
    disc_acc = D_acc_np(ys, disc_pred)

#     disc_epoch_size = ndisc(iter) * args.batch_size
#     while (disc_acc < 90) and (disc_iter < 1):
#         disc_iter += 1
#         x_true = np.concatenate([x_true_flow.next() for i in range(ndisc(iter))], axis=0)
#         x_predicted = generator.predict([np.random.normal(size=(disc_epoch_size, args.latent_dim))], batch_size=args.batch_size)
#         y_true = np.ones(x_true.shape[0]).reshape((-1,1)).astype("float32")
#         y_predicted = -np.ones(x_true.shape[0]).reshape((-1,1)).astype("float32")
#         xs = np.concatenate((x_predicted, x_true), axis=0)
#         ys = np.concatenate((y_predicted, y_true), axis=0)

#         discriminator.trainable = True
#         clipper.on_batch_begin()
#         disc_loss1 = discriminator.train_on_batch(x_true, y_true)
#         disc_loss2 = discriminator.train_on_batch(x_predicted, y_predicted)
#         disc_loss = (disc_loss1[0] + disc_loss2[0]) / 2

# #        disc_r = discriminator.fit(xs, ys, verbose=args.verbose, batch_size=args.batch_size, nb_epoch=1, shuffle=True, callbacks=callbacks)
# #        disc_loss = disc_r.history["loss"][0]
#         disc_pred = discriminator.predict(xs, batch_size=args.batch_size)
#         disc_acc = D_acc_np(ys, disc_pred)

    # update generator
    discriminator.trainable = False
    gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
    gen_loss = gen_disc.train_on_batch(gen_in, y_true)
    gen_pred = gen_disc.predict(gen_in, batch_size=args.batch_size)
    gen_acc = D_acc_np(y_true, gen_pred)

    print "Iter: {}, Generator: {} - {:.1f}%, Discriminator: {} - {:.1f}%".format(iter, gen_loss, gen_acc, disc_loss, disc_acc)
    if (iter+1) % args.frequency == 0:
        currTime = time.clock()
        elapsed = currTime - startTime
        second = elapsed % 60
        minute = int(elapsed / 60)
        print "Elapsed time: {}:{:.0f}".format(minute, second)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random-{}".format(args.prefix, iter+1), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        vis.saveModel(discriminator, args.prefix + "-disciminator")
        vis.saveModel(generator, args.prefix + "-generator")
        vis.saveModel(gen_disc, args.prefix + "-gendisc")

