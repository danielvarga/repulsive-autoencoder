import keras.backend as K
from keras import objectives
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda, Reshape, UpSampling2D, merge, Flatten
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
encoder_channels = model_dcgan.default_channels("encoder", "small", args.latent_dim)
generator_channels = model_dcgan.default_channels("generator", "small", args.original_shape[2])

reduction = 2 ** (len(generator_channels)+1)
assert args.original_shape[0] % reduction == 0
assert args.original_shape[1] % reduction == 0

gen_firstX = args.original_shape[0] // reduction
gen_firstY = args.original_shape[1] // reduction

#disc_layers = model_dcgan.discriminator_layers_dense(wd=args.encoder_wd, bn_allowed=args.decoder_use_bn)
#disc_layers = model_dcgan.discriminator_layers_wgan([16, 32, 64, 128], wd=args.encoder_wd, bn_allowed=True)

gen_layers = model_dcgan.generator_layers_wgan(generator_channels, args.latent_dim, args.decoder_wd, args.decoder_use_bn, args.batch_size, gen_firstX, gen_firstY)
enc_layers = model_dcgan.encoder_layers_wgan(encoder_channels, args.encoder_wd, args.encoder_use_bn)

#gen_layers = model_dcgan.generator_layers_simple(args.latent_dim, True, 0.0, True, args.color, [64])
#enc_layers = model_dcgan.encoder_layers_simple(args.latent_dim, True, 0.0, True, args.color, [64])

enc_input = Input(batch_shape=([args.batch_size] + list(args.original_shape)), name="enc_input")
gen_input = Input(batch_shape=(args.batch_size, args.latent_dim), name="gen_input")


pair_outs = []
enc_outs = []
enc_output = enc_input
for layer in enc_layers:
    enc_output = layer(enc_output)
    enc_outs.append(enc_output)

gen_outs = []
ae_output = enc_output
ae_z = enc_output
gen_output = gen_input
for layer in gen_layers:
    ae_output = layer(ae_output)
    gen_output = layer(gen_output)
    gen_outs.append(ae_output)

# Put on the critics

pair_outs = []
for enc_out in enc_outs:
    if enc_out.name.lower().startswith("relu"):
	for gen_out in gen_outs:
	    if gen_out.name.lower().startswith("relu") and gen_out.shape == enc_out.shape:
	        pair_outs.append((enc_out, gen_out))


print(pair_outs)
print(enc_outs)
print(gen_outs)

import math
critics_data = [(enc_input, gen_output, "fokritik")]
disc_inputs = []
for critic_data in critics_data:
    real_o, gen_o, critic_name = critic_data
    
    #batch_shape = [args.batch_size] + real_layer.shape.as_list()
    batch_shape = real_o.shape.as_list()
    #print(batch_shape)

    disc_input = Input(batch_shape=batch_shape, name=critic_name+"_input")
    disc_input_latent = Input(batch_shape=(args.batch_size, args.latent_dim), name=critic_name+"_input_latent")

    disc_inputs.append(disc_input)
    
    filter_nums = []
    for i in range(int(math.log(batch_shape[1],2))-2):
	filter_nums.append(32*(2**i))
    filter_nums.append(1)

    print(filter_nums)
    disc_layers = model_dcgan.discriminator_layers_wgan(filter_nums, wd=args.encoder_wd, bn_allowed=True)
    #disc_layers = model_dcgan.discriminator_layers_dense(0.0, True)
    #disc_layers = [Flatten()] + disc_layers
    disc_layers_latent = model_dcgan.discriminator_layers_dense(0.0, True)
    print(disc_layers)

    print(gen_output.shape.as_list())
    gen_disc_output = gen_o
    

    gen_disc_latent_output = enc_output
    disc_output_latent = disc_input_latent

    """
    disc_layers_latent = []
    disc_layers_latent.append(Reshape((1,1,args.latent_dim)))
    disc_layers_latent.append(UpSampling2D(size=(64,64)))

    for layer in disc_layers_latent:
        gen_disc_latent_output = layer(gen_disc_latent_output)
        disc_output_latent = layer(disc_output_latent)
    """


    disc_output = disc_input

    """
    disc_output = merge([disc_output, disc_output_latent], mode='concat', concat_axis=-1)
    gen_disc_output = merge([gen_disc_output, gen_disc_latent_output], mode='concat', concat_axis=-1)
    """

    for layer in disc_layers:
        gen_disc_output = layer(gen_disc_output)
        disc_output = layer(disc_output)

    gen_disc_latent_output = enc_output
    disc_output_latent = disc_input_latent
    for layer in disc_layers_latent:
        gen_disc_latent_output = layer(gen_disc_latent_output)
        disc_output_latent = layer(disc_output_latent)

    disc_output = Lambda(lambda x: x[0]+x[1])([disc_output_latent,disc_output])
    gen_disc_output = Lambda(lambda x: x[0]+x[1])([gen_disc_latent_output,gen_disc_output])


def mse_loss(x, x_decoded):
    return K.mean(objectives.mean_squared_error(x, x_decoded))

def critic_loss(x, x_decoded):
    return - K.mean(gen_disc_output)

def ae_loss(x, x_decoded):
    #pair = pair_outs[1]
    #return mse_loss(pair[0], pair[1])

    summ = 0.0 * mse_loss(x, x_decoded)
    for pair in pair_outs:
	summ += mse_loss(K.flatten(pair[0]), K.flatten(pair[1]))
    return summ
    return mse_loss(x, x_decoded) + summ # + K.mean(K.square(ae_z)) # + critic_loss(x, x_decoded)

    z = ae_z
    z_centered = z - K.mean(z, axis=0)
    cov = K.dot(K.transpose(z_centered), z_centered) / args.batch_size
    # K.mean(tf.diag_part(cov))
    loss = K.mean(K.square(K.eye(200) - cov))# * (args.batch_size ** 2)
    return mse_loss(x, x_decoded) + loss

def covariance_monitor(x, x_decoded):
    z = enc_output
    z_centered = z - K.mean(z, axis=0)
    cov = K.dot(K.transpose(z_centered), z_centered) / args.batch_size
    return K.mean(tf.diag_part(cov))

encoder = Model(enc_input, enc_output)
generator = Model(gen_input, gen_output)
ae = Model(enc_input, ae_output)

gen_disc = Model([gen_input, enc_input], gen_disc_output)
disc = Model([disc_input, disc_input_latent], disc_output)

if args.optimizer == "adam":
    ae_optimizer = Adam(lr=args.lr)
    gen_disc_optimizer = Adam(lr=args.lr)
    disc_optimizer = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    ae_optimizer = RMSprop(lr=args.lr)
    gen_disc_optimizer = RMSprop(lr=args.lr)
    disc_optimizer = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    ae_optimizer = SGD(lr=args.lr)
    gen_disc_optimizer = SGD(lr=args.lr)
    disc_optimizer = SGD(lr=args.lr)

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    return - y_true * y_pred

# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


generator.compile(optimizer=disc_optimizer, loss = "mse")

disc.compile(optimizer=disc_optimizer, loss = D_loss)
make_trainable(disc, False)
gen_disc.compile(optimizer=gen_disc_optimizer, loss = D_loss)
#make_trainable(encoder, False) # TODO this is not  going to work as the inference model is not guided at all by the reconstruction

#make_trainable(generator, False)
ae.compile(optimizer=ae_optimizer, loss=ae_loss, metrics = [mse_loss, covariance_monitor])
#make_trainable(generator, True)

#make_trainable(disc, True)

print "Discriminator"
disc.summary()
print "AE"
ae.summary()
print "Enc_Disc"
gen_disc.summary()

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

    def on_batch_begin(self, x, y):
        self.clip()

    def clip(self):
	for layer in self.layers:
#            if layer.__class__.__name__ not in ("Convolution2D", "BatchNormalization"): continue
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = np.clip(weights[i], - self.clipValue, self.clipValue)
            layer.set_weights(weights)
clipper = ClipperCallback(disc_layers + disc_layers_latent, args.clipValue)

y_generated = np.array([-1.0] * args.batch_size).reshape((-1,1)).astype("float32")
y_gaussian = np.array([1.0] * args.batch_size).reshape((-1,1)).astype("float32")

print "starting training"
startTime = time.clock()
for iter in range(args.nb_epoch):

    # update autoencoder
    """
    recons_iters = nrecons(iter)
    images = np.concatenate([x_true_flow.next() for i in range(recons_iters)], axis=0)
    r = ae.fit(images, images, verbose=args.verbose, batch_size=args.batch_size, nb_epoch=1)
    recons_loss = r.history["loss"][0]
    cov_monitor = r.history["covariance_monitor"][0]
    mse_monitor = np.float32(np.prod(args.original_shape)) * r.history["mse_loss"][0]
    critic_monitor = r.history["critic_loss"][0]
    """
    # update generator
    rnd = np.random.normal(size=(args.batch_size, args.latent_dim))
    #r = gen_disc.fit(rnd, - np.ones(args.batch_size), verbose=args.verbose, batch_size=args.batch_size, nb_epoch=1)
    disc.trainable = False
    r = gen_disc.train_on_batch([rnd, x_true_flow.next()], np.ones(args.batch_size))
    disc.trainable = True
    enc_loss = r

    # update discriminator
    disc_iters = ndisc(iter)

    """
    images = np.concatenate([x_true_flow.next() for i in range(disc_iters)], axis=0)

    rnd = np.random.normal(size=(disc_iters * args.batch_size, args.latent_dim))
    x_generated = generator.predict(rnd, batch_size=args.batch_size)
    x_true = images

    xs = np.concatenate((x_generated, x_true), axis=0)
    ys = np.concatenate((np.ones((args.batch_size * disc_iters)), - np.ones((args.batch_size * disc_iters))), axis=0).reshape((-1,1)).astype("float32")
    clipper.clip()
    """

    for i in range(disc_iters):
        clipper.clip()
        #r = disc.fit(xs, ys, verbose=args.verbose, batch_size=args.batch_size, shuffle=False, nb_epoch=1, callbacks=[clipper])
	bs = args.batch_size

        rnd = np.random.normal(size=(args.batch_size, args.latent_dim))

	x_generated = generator.predict(rnd, batch_size=args.batch_size)
	x_true = x_true_flow.next()
	x_encoded   = encoder.predict(x_true, batch_size=args.batch_size)
	x_true_pairs = [x_true, x_encoded]
	x_fake_pairs = [x_generated, rnd]

	r = disc.train_on_batch(x_fake_pairs, -np.ones(bs))
	r = disc.train_on_batch(x_true_pairs, np.ones(bs))
        #disc_loss = r.history["loss"][0]
	disc_loss = r

    #images = [x_true_flow.next() for i in range(disc_iters)]
    images = x_true_flow.next()
    r = ae.fit(images, images, verbose=args.verbose, batch_size=args.batch_size, nb_epoch=1)
    cov_monitor = r.history['covariance_monitor'][0]

    # # update encoder
    # image_batch = x_true_flow.next()
    # enc_loss = enc_disc.train_on_batch(image_batch, y_gaussian)
    #enc_loss = 0

    #print "Iter: {}, Disc: {}, Enc: {}, Ae: {}, Cov: {}, Mse: {}, Critic: {}".format(iter+1, disc_loss, enc_loss, recons_loss, cov_monitor, mse_monitor, critic_monitor)
    print "Iter: {}, Disc: {}, Enc: {}, Cov: {}".format(iter+1, disc_loss, enc_loss, cov_monitor)

    #print "Iter: {}, Disc: {}, Enc: {}".format(iter+1, disc_loss, enc_loss)
    if (iter+1) % args.frequency == 0:
        currTime = time.clock()
        elapsed = currTime - startTime
        second = elapsed % 60
        minute = int(elapsed / 60)
        print "Elapsed time: {}:{:.0f}".format(minute, second)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random-{}".format(args.prefix, iter+1), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, gaussian_sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        if (iter+1) % 200 == 0: 
            vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist-{}.png".format(args.prefix, iter+1))
            vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
            vis.displaySet(x_test[:args.batch_size], args.batch_size, args.batch_size, ae, "%s-test" % args.prefix)
            vis.displaySet(x_train[:args.batch_size], args.batch_size, args.batch_size, ae, "%s-train" % args.prefix)
            vis.saveModel(disc, args.prefix + "-discriminator")
            vis.saveModel(encoder, args.prefix + "-encoder")
            vis.saveModel(generator, args.prefix + "-generator")
            vis.saveModel(gen_disc, args.prefix + "-gen_disc")

