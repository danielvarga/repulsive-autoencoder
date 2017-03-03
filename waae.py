import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.optimizers import *
import numpy as np
import time
import params
import vis


args = params.getArgs()
args.clipValue = 0.01 #TODO
print(args)

# limit memory usage
import keras
print "Keras version: ", keras.__version__
if keras.backend._BACKEND == "tensorflow":
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))

import data
(x_train, x_test) = data.load(args.dataset, args.trainSize, args.testSize, shape=args.shape, color=args.color)
args.original_shape = x_train.shape[1:]
imageGenerator = ImageDataGenerator()
x_train_size = args.batch_size * (x_train.shape[0] // args.batch_size)
x_train = x_train[:x_train_size]
print "Train set size: ", x_train_size 
x_true_flow = imageGenerator.flow(x_train, batch_size = args.batch_size)


import model_dcgan
disc_layers = model_dcgan.discriminator_layers_dense(wd=args.encoder_wd, bn_allowed=args.decoder_use_bn)
gen_layers = model_dcgan.generator_layers_wgan(latent_dim=args.latent_dim, batch_size=args.batch_size, wd=args.decoder_wd, bn_allowed=args.decoder_use_bn, image_channel=x_train.shape[3])
enc_layers = model_dcgan.encoder_layers_wgan(latent_dim=args.latent_dim, batch_size=args.batch_size, wd=args.encoder_wd, bn_allowed=args.decoder_use_bn, image_channel=x_train.shape[3])

enc_input = Input(batch_shape=([args.batch_size] + list(args.original_shape)), name="enc_input")
disc_input = Input(batch_shape=(args.batch_size, args.latent_dim), name="disc_input")

enc_output = enc_input
for layer in enc_layers:
    enc_output = layer(enc_output)
ae_output = enc_output
gen_output = disc_input
for layer in gen_layers:
    ae_output = layer(ae_output)
    gen_output = layer(gen_output)
enc_disc_output = enc_output
disc_output = disc_input
for layer in disc_layers:
    enc_disc_output = layer(enc_disc_output)
    disc_output = layer(disc_output)

encoder = Model(enc_input, enc_output)
generator = Model(disc_input, gen_output)
ae = Model(enc_input, ae_output)
enc_disc = Model(enc_input, enc_disc_output)
disc = Model(disc_input, disc_output)
if args.optimizer == "adam":
    ae_optimizer = Adam(lr=args.lr)
    enc_disc_optimizer = Adam(lr=args.lr)
    disc_optimizer = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    ae_optimizer = RMSprop(lr=args.lr)
    enc_disc_optimizer = RMSprop(lr=args.lr)
    disc_optimizer = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    ae_optimizer = SGD(lr=args.lr)
    enc_disc_optimizer = SGD(lr=args.lr)
    disc_optimizer = SGD(lr=args.lr)

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    return - K.mean(y_true * y_pred)

ae.compile(optimizer=ae_optimizer, loss="mse")
enc_disc.compile(optimizer=enc_disc_optimizer, loss = D_loss)
disc.compile(optimizer=disc_optimizer, loss = D_loss)

print "Discriminator"
disc.summary()
print "AE"
ae.summary()
print "Enc_Disc"
enc_disc.summary()

def ndisc(gen_iters):
    if gen_iters < 25:
        return 100
    elif gen_iters % 500 == 0:
	return 100
    else:
        return 5

def nrecons(gen_iters):
    return 1

def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))

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
clipper = ClipperCallback(disc_layers, args.clipValue)

y_generated = np.array([-1.0] *  args.batch_size).reshape((-1,1)).astype("float32")
y_gaussian = np.array([1.0] *  args.batch_size).reshape((-1,1)).astype("float32")

print "starting training"
startTime = time.clock()
for iter in range(args.nb_epoch):

    # update autoencoder
#    encoder.trainable = False # TODO does it make sense to freeze the encoder when minimizing reconstruction loss?
    recons_iters = nrecons(iter)
    images = np.concatenate([x_true_flow.next() for i in range(recons_iters)], axis=0)
    r = ae.fit(images, images, verbose=args.verbose, batch_size=args.batch_size, nb_epoch=1)
    recons_loss = r.history["loss"][0]

    # update discriminator
    disc.trainable = True
    disc_iters = ndisc(iter)
    images = np.concatenate([x_true_flow.next() for i in range(disc_iters)], axis=0)
    x_generated = encoder.predict(images, batch_size=args.batch_size)
    x_gaussian = np.random.normal(size=(disc_iters * args.batch_size, args.latent_dim))
    xs = np.concatenate((x_generated, x_gaussian), axis=0)
    ys = np.concatenate((-1 * np.ones((args.batch_size * disc_iters)), np.ones((args.batch_size * disc_iters))), axis=0).reshape((-1,1)).astype("float32")
    clipper.clip()
    r = disc.fit(xs, ys, verbose=args.verbose, batch_size=args.batch_size, shuffle=True, nb_epoch=1)
    disc_loss = r.history["loss"][0]

    # update encoder
    encoder.trainable = True
    disc.trainable = False
    image_batch = x_true_flow.next()
    enc_loss = enc_disc.train_on_batch(image_batch, y_gaussian)

    print "Iter: {}, Disc: {}, Enc: {}, Recons: {}".format(iter+1, disc_loss, enc_loss, recons_loss)
    if (iter+1) % args.frequency == 0:
        currTime = time.clock()
        elapsed = currTime - startTime
        second = elapsed % 60
        minute = int(elapsed / 60)
        print "Elapsed time: {}:{:.0f}".format(minute, second)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random-{}".format(args.prefix, iter+1), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        vis.displaySet(x_test[:args.batch_size], args.batch_size, 100, ae, "%s-test" % args.prefix)
        vis.displaySet(x_train[:args.batch_size], args.batch_size, 100, ae, "%s-train" % args.prefix)
        vis.saveModel(disc, args.prefix + "-discriminator")
        vis.saveModel(encoder, args.prefix + "-encoder")
        vis.saveModel(generator, args.prefix + "-generator")
        vis.saveModel(enc_disc, args.prefix + "-enc_disc")

