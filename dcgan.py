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
x_train_size = args.batch_size * (x_train.shape[0] // args.batch_size)
x_train = x_train[:x_train_size]
args.original_shape = x_train.shape[1:]
print "Train set size: ", x_train_size 
x_true_flow = imageGenerator.flow(x_train, batch_size = args.batch_size)


print "building networks"
generator_channels = model_dcgan.default_channels("generator", args.gen_size, args.original_shape[2])
discriminator_channels = model_dcgan.default_channels("discriminator", args.disc_size, None)

reduction = 2 ** (len(generator_channels)+1)
assert args.original_shape[0] % reduction == 0
assert args.original_shape[1] % reduction == 0
gen_firstX = args.original_shape[0] // reduction
gen_firstY = args.original_shape[1] // reduction

gen_layers = model_dcgan.generator_layers_wgan(generator_channels, args.latent_dim, args.wd, args.use_bn_gen, args.batch_size, gen_firstX, gen_firstY)
disc_layers = model_dcgan.discriminator_layers_wgan(discriminator_channels, wd=args.wd, bn_allowed=args.use_bn_disc)

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
    return - K.mean(y_true * y_pred)

def D_acc(y_true, y_pred):
    x = y_true * y_pred
    return 100 * K.mean(x > 0)
def D_acc_np(y_true, y_pred):
    x = y_true * y_pred
    return 100 * np.mean(x > 0)

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val



discriminator = Model(disc_input, disc_output)
if args.optimizer == "adam":
    optimizer_d = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_d = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    optimizer_d = SGD(lr=args.lr)
make_trainable(discriminator, True)
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
make_trainable(discriminator, False)
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

test_true = x_test[:args.batch_size]
test_gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
def evaluate():
    test_generated = generator.predict(test_gen_in, batch_size=args.batch_size)
    test_x = np.concatenate((test_generated, test_true), axis=0)
    test_y = ys
    pred = discriminator.predict(test_x, batch_size=args.batch_size)
    divergence = np.mean(pred * test_y)
    return divergence

print "starting training"
startTime = time.clock()
for iter in range(args.nb_iter):
    # update discriminator
    disc_iters = ndisc(iter)
    if False:
        x_true = np.concatenate([x_true_flow.next() for i in range(disc_iters)], axis=0)
        gen_in = np.random.normal(size=(args.batch_size * disc_iters, args.latent_dim))
        x_generated = generator.predict(gen_in, batch_size=args.batch_size)
        xs = np.concatenate((x_generated, x_true), axis=0)
        ys = np.concatenate((-1 * np.ones((args.batch_size * disc_iters)), np.ones((args.batch_size * disc_iters))), axis=0).reshape((-1,1)).astype("float32")
        clipper.clip()
        r = discriminator.fit(xs, ys, verbose=args.verbose, batch_size=args.batch_size, shuffle=True, nb_epoch=1)
        disc_loss = r.history["loss"][0]
    else:
        for disc_iter in range(disc_iters):
            x_true = x_true_flow.next()
            gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
            x_generated = generator.predict(gen_in, batch_size=args.batch_size)
            clipper.clip()
            disc_loss1 = discriminator.train_on_batch(x_true, y_true)
            disc_loss2 = discriminator.train_on_batch(x_generated, y_generated)
        disc_loss = disc_loss1[0] + disc_loss2[0]
        xs = np.concatenate((x_generated, x_true), axis=0)

    disc_eval = evaluate()
    #disc_pred = discriminator.predict(xs, batch_size=args.batch_size)
    #disc_acc = D_acc_np(ys, disc_pred)

    # update generator
#    make_trainable(discriminator, False)
    gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
    gen_loss = gen_disc.train_on_batch(gen_in, y_true)
    gen_eval = evaluate()
    #gen_pred = gen_disc.predict(gen_in, batch_size=args.batch_size)
    #gen_acc = D_acc_np(y_true, gen_pred)

    print "Iter: {}, Generator: {} - {:.3f}, Discriminator: {} - {:.3f}".format(iter, gen_loss, gen_eval, disc_loss, disc_eval)
    if (iter+1) % args.frequency == 0:
        currTime = time.clock()
        elapsed = currTime - startTime
        second = elapsed % 60
        minute = int(elapsed / 60)
        print "Elapsed time: {}:{:.0f}".format(minute, second)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random-{}".format(args.prefix, iter+1), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        vis.saveModel(discriminator, args.prefix + "_discriminator")
        vis.saveModel(generator, args.prefix + "_generator")
        vis.saveModel(gen_disc, args.prefix + "_gendisc")

