'''This AE differs from a standard VAE (https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
in the following ways:

- The latent variances are not learned, they are simply set to 0.
- A normalization step takes the latent variables to a sphere surface.
- The regularization loss is changed to an energy term that
  corresponds to a pairwise repulsive force between the encoded
  elements of the minibatch.
'''

import params
args = params.getArgs()
print(args)

# limit memory usage
import keras
if keras.backend._BACKEND == "tensorflow":
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))

import data
(x_train, x_test) = data.load(args.dataset, args.trainSize, args.testSize, shape=args.shape, color=args.color)
args.original_shape = x_train.shape[1:]

import model
ae, encoder, encoder_var, generator = model.build_model(args)
ae.summary()

if args.spherical:
    sampler = model.spherical_sampler
else:
    sampler = model.gaussian_sampler

import callbacks
cbs = [callbacks.FlushCallback()]
cbs.append(callbacks.get_lr_scheduler(args.nb_epoch, args.lr))
cbs.append(callbacks.ImageDisplayCallback(
    x_train, x_test,
    args.latent_dim, args.batch_size,
    encoder, encoder_var, generator, sampler,
    args.callback_prefix, args.frequency))
for schedule in args.weight_schedules:
    if schedule[1] != schedule[2]:
        cbs.append(callbacks.WeightSchedulerCallback(args.nb_epoch, schedule[0], schedule[1], schedule[2], schedule[3], schedule[4], schedule[5]))


ae.fit(x_train, x_train,
       verbose=args.verbose,
       shuffle=True,
       nb_epoch=args.nb_epoch,
       batch_size=args.batch_size,
       callbacks = cbs,
       validation_data=(x_test, x_test))


import vis
vis.saveModel(ae, args.prefix + "_model")
vis.saveModel(encoder, args.prefix + "_encoder")
vis.saveModel(encoder_var, args.prefix + "_encoder_var")
vis.saveModel(generator, args.prefix + "_generator")

# display randomly generated images
vis.displayRandom(15, x_test, args.latent_dim, sampler, generator, "%s-random" % args.prefix, batch_size=args.batch_size)

vis.displaySet(x_test[:args.batch_size], args.batch_size, 100, ae, "%s-test" % args.prefix)
vis.displaySet(x_train[:args.batch_size], args.batch_size, 100, ae, "%s-train" % args.prefix)

# display image interpolation
vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, generator, 10, "%s-interp" % args.prefix)

vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))


# # display a 2D plot of the validation set in the latent space
# vis.latentScatter(encoder, x_test, batch_size, args.prefix+"-fig1")

# display 2D manifolds of images
# show_manifolds = False
# if show_manifolds:
#     for y in range(1, args.latent_dim-1):
#         vis.displayImageManifold(30, args.latent_dim, generator, height, width, 0, y, y+1, "%s-manifold%d" % (args.prefix, y), batch_size=batch_size)
