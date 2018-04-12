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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils

import keras.backend as K
import tensorflow as tf
import time

import params
import data
import vis
import load_models
import model_dcgan

import callbacks
import samplers

args = params.getArgs()
print(args)

# set random seed
np.random.seed(10)


# limit memory usage
import keras
print("Keras version: ", keras.__version__)
if keras.backend._BACKEND == "tensorflow":
    import tensorflow as tf
    print("Tensorflow version: ", tf.__version__)
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))


epsilon = 0.00001

############################################
print("loading data")

data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
x_true_flow = data_object.get_train_flow(args.batch_size)
args.original_shape = x_train.shape[1:]

############################################
print("specify loss functions")

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    return - K.mean(y_true * y_pred)

def grad_aux(output, input):
    grads = K.gradients(output, [input])
    grads = grads[0]
    tensor_axes = list(range(1, K.ndim(grads)))
    grads = K.sqrt(epsilon + K.sum(K.square(grads), axis=tensor_axes))
    return grads

def grad_loss(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.square(K.maximum(k1, grads) - k1))
    return grad_penalty

def grad_loss_orig(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.square(grads - k1))
    return grad_penalty


############################################
print("auxiliary functions")

def display_elapsed(startTime, endTime):
    elapsed = endTime - startTime
    second = elapsed % 60
    minute = int(elapsed / 60)
    print("Elapsed time: {}:{:.0f}".format(minute, second))
# Freeze weights in the discriminator for stacked training
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
def ndisc(gen_iters):
    if gen_iters <= 25:
        return 100
    elif gen_iters % 500 == 0:
        return 100
    else:
        return 5

def restart_disc(gen_iters):
    return False
    if (gen_iters) in (500, 1000, 2000):
        return True
    else:
        return False

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

############################################
print("building networks")

generator_channels = model_dcgan.default_channels("generator", args.dcgan_size, args.original_shape[2])
discriminator_channels = model_dcgan.default_channels("discriminator", args.disc_size, None)


if args.generator == "dcgan":
    reduction = 2 ** (len(generator_channels)+1)
    assert args.original_shape[0] % reduction == 0
    assert args.original_shape[1] % reduction == 0
    gen_firstX = args.original_shape[0] // reduction
    gen_firstY = args.original_shape[1] // reduction
    gen_layers = model_dcgan.generator_layers_wgan(generator_channels, args.latent_dim, args.generator_wd, args.use_bn_gen, args.batch_size, gen_firstX, gen_firstY)
elif args.generator == "dense":
    gen_layers = model_dcgan.generator_layers_dense(args.latent_dim, args.batch_size, args.generator_wd, 
            args.use_bn_gen, args.original_shape, args.intermediate_dims)
else:
    assert False, "Invalid generator type"

if args.discriminator =="dcgan":
    disc_layers = model_dcgan.discriminator_layers_wgan(discriminator_channels, wd=args.discriminator_wd, bn_allowed=args.use_bn_disc)
elif args.discriminator == "dense":
    disc_layers = model_dcgan.discriminator_layers_dense(args.discriminator_wd, args.use_bn_disc)
else:
    assert False, "Invalid discriminator type"

gen_input = Input(batch_shape=(args.batch_size,args.latent_dim), name="gen_input")
disc_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name="disc_input")

if args.optimizer == "adam":
    optimizer_d = Adam(lr=args.lr)
    optimizer_g = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_d = RMSprop(lr=args.lr)
    optimizer_g = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    optimizer_d = SGD(lr=args.lr)
    optimizer_g = SGD(lr=args.lr)

sampler = samplers.gaussian_sampler
                    

def build_networks(gen_layers, disc_layers):
    gen_output = gen_input
    disc_output = disc_input
    gen_disc_output = gen_output

    for layer in gen_layers:
        gen_output = layer(gen_output)
        gen_disc_output = layer(gen_disc_output)

    for layer in disc_layers:
        disc_output = layer(disc_output)
        gen_disc_output = layer(gen_disc_output)

    generator = Model(inputs=gen_input, outputs=gen_output)
    gen_disc = Model(inputs=gen_input, outputs=gen_disc_output)
    discriminator = Model(inputs=disc_input, outputs=disc_output)
    discriminator_grad = Model(inputs=disc_input, outputs=disc_output)

    # callback for clipping weights
    clipper = callbacks.ClipperCallback(discriminator.layers, args.clipValue)

    # callback for saving generated images
    generated_saver = callbacks.SaveGeneratedCallback(generator, sampler, args.prefix, args.batch_size, args.frequency, args.latent_dim)

    make_trainable(discriminator, True)
    discriminator.compile(optimizer=optimizer_d, loss=D_loss)
    make_trainable(discriminator_grad, True)
    discriminator_grad.compile(optimizer=optimizer_d, loss=grad_loss)
    generator.compile(optimizer=optimizer_g, loss="mse")
    make_trainable(discriminator, False)
    gen_disc.compile(loss=D_loss, optimizer=optimizer_g)

    return (generator, discriminator, discriminator_grad, gen_disc, clipper, generated_saver)

if args.modelPath is None:
    (generator, discriminator, discriminator_grad, gen_disc, clipper, generated_saver) = build_networks(gen_layers, disc_layers)
else:
    print("Loading models from " +args.modelPath)
    generator = load_models.loadModel(args.modelPath + "_generator")
    discriminator = load_models.loadModel(args.modelPath + "_discriminator")
    discriminator_grad = load_models.loadModel(args.modelPath + "_discriminator_grad")
    gen_disc = load_models.loadModel(args.modelPath + "_gendisc")
    clipper = callbacks.ClipperCallback(discriminator.layers, args.clipValue)
    make_trainable(discriminator, True)
    discriminator.compile(optimizer=optimizer_d, loss=D_loss)
    make_trainable(discriminator_grad, True)
    if args.gradient_penalty == "grad":
        discriminator_grad.compile(optimizer=optimizer_d, loss=grad_loss)
    elif args.gradient_penalty == "grad_orig":
        discriminator_grad.compile(optimizer=optimizer_d, loss=grad_orig_loss)
    generator.compile(optimizer=optimizer_g, loss="mse")
    make_trainable(discriminator, False)
    gen_disc.compile(loss=D_loss, optimizer=optimizer_g)

print("Discriminator")
discriminator.summary()
print("Generator:")
generator.summary()

############################################
print("y values for evaluation")
y_generated = np.array([-1.0] *  args.batch_size).reshape((-1,1)).astype("float32")
y_true = np.array([1.0] *  args.batch_size).reshape((-1,1)).astype("float32")
    

############################################
print("starting training")
vis.plotImages(x_train[:100], 10, 10, args.prefix + "-orig")
disc_offset = 0
startTime = time.clock()

if args.gradient_penalty != "no" and args.clipValue > 0:
    print("!!!!!! WARNING: both clipping and gradient penality enabled. Are you sure this is intended?")

for iter in range(1, args.nb_iter+1):
    # update discriminator
    disc_iters = ndisc(iter - disc_offset)
    if False: # TODO decide which branch is the good one
        x_true = np.concatenate([x_true_flow.next()[0] for i in range(disc_iters)], axis=0)
        gen_in = np.random.normal(size=(args.batch_size * disc_iters, args.latent_dim))
        x_generated = generator.predict(gen_in, batch_size=args.batch_size)
        xs = np.concatenate((x_generated, x_true), axis=0)
        ys = np.concatenate((-1 * np.ones((args.batch_size * disc_iters)), np.ones((args.batch_size * disc_iters))), axis=0).reshape((-1,1)).astype("float32")
        
        r = discriminator.fit(xs, ys, verbose=args.verbose, batch_size=args.batch_size, shuffle=True, epochs=1)
        clipper.clip()
        disc_loss = r.history["loss"][0]
        grad_loss = r.history["grad_loss"][0]
    else:
        disc_loss = 0
        grad_loss = 0
        for disc_iter in range(disc_iters):
            x_true = x_true_flow.next()[0]
            gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
            x_generated = generator.predict(gen_in, batch_size=args.batch_size)

            disc_loss1 = discriminator.train_on_batch(x_true, y_true)
            disc_loss2 = discriminator.train_on_batch(x_generated, y_generated)

            if args.gradient_penalty != "no":
                weights = np.random.uniform(size=x_true.shape)
                interp_points = x_true * weights + x_generated * (1-weights)
                grad_loss_curr = discriminator_grad.train_on_batch(interp_points, y_true)
            else:
                grad_loss_curr = 0
            clipper.clip()
            
            disc_loss += disc_loss1 + disc_loss2
            grad_loss += grad_loss_curr
        disc_loss /= disc_iters
        grad_loss /= disc_iters

    # update generator
#    make_trainable(discriminator, False)
    gen_in = np.random.normal(size=(args.batch_size, args.latent_dim))
    gen_loss = gen_disc.train_on_batch(gen_in, y_true)

    print("Iter: {}, Discriminator: {}, Generator: {}, grad: {}".format(iter, disc_loss, gen_loss, grad_loss))

    # syn-constant-uniform specific: print average variance of images to monitor the spottedness of the images
    if False and args.dataset == 'syn-constant-uniform':
        vr = np.average( np.var(x_generated, axis=(1,2)) )
        with open(args.prefix + '_vars.txt', 'a') as f:
            f.write(str(vr) + '\n')
        #print ( np.var(x_generated, axis=(1,2)).shape )

    if iter % args.frequency == 0:
        currTime = time.clock()
        display_elapsed(startTime, currTime)
        vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random-{}".format(args.prefix, iter), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        latent_samples = np.random.normal(size=(2, args.latent_dim))
        vis.interpBetween(latent_samples[0], latent_samples[1], generator, args.batch_size, args.prefix + "_interpBetween-{}".format(iter))
        vis.interpBetween(latent_samples[0], latent_samples[1], generator, args.batch_size, args.prefix + "_interpBetween")
        load_models.saveModel(discriminator, args.prefix + "_discriminator")
        load_models.saveModel(discriminator_grad, args.prefix + "_discriminator_grad")
        load_models.saveModel(generator, args.prefix + "_generator")
        load_models.saveModel(gen_disc, args.prefix + "_gendisc")
    if iter % (args.nb_iter // 10) == 0:        
        load_models.saveModel(discriminator, args.prefix + "_discriminator_{}".format(iter))
        load_models.saveModel(discriminator_grad, args.prefix + "_discriminator_grad_{}".format(iter))
        load_models.saveModel(generator, args.prefix + "_generator_{}".format(iter))
        load_models.saveModel(gen_disc, args.prefix + "_gendisc_{}".format(iter))
        generated_saver.save(iter)

    if restart_disc(iter): # restart discriminator
        print("Restarting discriminator!!!!!!!!!")
        disc_offset = iter
        load_models.saveModel(discriminator, args.prefix + "_discriminator_restarted_{}".format(iter))
        if args.discriminator =="dcgan":
            disc_layers = model_dcgan.discriminator_layers_wgan(discriminator_channels, wd=args.discriminator_wd, bn_allowed=args.use_bn_disc)
        elif args.discriminator == "dense":
            disc_layers = model_dcgan.discriminator_layers_dense(args.discriminator_wd, args.use_bn_disc)
        else:
            assert False, "Invalid discriminator type"
        (generator, discriminator, discriminator_grad, gen_disc, clipper, generated_saver) = build_networks(gen_layers, disc_layers)


############################################
print("dead code chunks")


# def update_lr(gen_iters):
#     phase = gen_iters * 1.0 / args.nb_iter
#     if phase == 0.0:
#         multiplier = 10
#     elif phase == 0.3:
#         multiplier = 1
#     elif phase == 0.6:
#         multiplier = 0.1
#     elif phase == 0.9:
#         multiplier = 0.01
#     else:
#         return
#     print "Setting generator learning rate to: ", args.lr * multiplier
#     K.set_value(optimizer_g.lr, args.lr * multiplier)

# sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
# K.set_value(sgd.lr, 0.5 * K.get_value(sgd.lr))


# count = 10 * args.batch_size
# test_true = x_test[:count]
# test_gen_in = np.random.normal(size=(count, args.latent_dim))
# def evaluate():
#     test_generated = generator.predict(test_gen_in, batch_size=args.batch_size)
# #    test_x = np.concatenate((test_generated, test_true), axis=0)
# #    test_y = ys
# #    pred = discriminator.predict(test_x, batch_size=args.batch_size)
# #    divergence = np.mean(pred * test_y)
#     emd = vis.dataset_emd(test_true, test_generated)
#     return emd
# eval_start_time = time.clock()
# initial_emd = evaluate()
# eval_end_time = time.clock()
# display_elapsed(eval_start_time, eval_end_time)
# print "initial emd: {}".format(initial_emd)
