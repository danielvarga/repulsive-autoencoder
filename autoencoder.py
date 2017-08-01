'''This AE differs from a standard VAE (https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
in the following ways:

- The latent variances are not learned, they are simply set to 0.
- A normalization step takes the latent variables to a sphere surface.
- The regularization loss is changed to an energy term that
  corresponds to a pairwise repulsive force between the encoded
  elements of the minibatch.
'''
import numpy as np
np.random.seed(10)

import tensorflow as tf
tf.set_random_seed(10)

from keras.callbacks import LearningRateScheduler
import keras.backend as K
import params
import vis
import load_models

args = params.getArgs()
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
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
args.original_shape = x_train.shape[1:]

import samplers
sampler = samplers.sampler_factory(args, x_train)

import model
modelDict = model.build_model(args)
ae = modelDict.ae
encoder = modelDict.encoder
encoder_var = modelDict.encoder_var
generator = modelDict.generator
ae.summary()


import callbacks
cbs = [callbacks.FlushCallback()]
get_lr = callbacks.get_lr_scheduler(args.nb_epoch, args.lr, args.lr_decay_schedule)
cbs.append(LearningRateScheduler(get_lr))
# cbs.append(callbacks.SaveGeneratedCallback(generator, sampler, args.prefix, args.batch_size, 20, args.latent_dim))
cbs.append(callbacks.ImageDisplayCallback(x_train, x_test, args, modelDict, sampler, data_object.anchor_indices))
cbs.append(callbacks.SaveModelsCallback(ae, encoder, encoder_var, generator, args.prefix, args.frequency))
for schedule in args.weight_schedules:
    if schedule[1] != schedule[2]:
        cbs.append(callbacks.WeightSchedulerCallback(args.nb_epoch, schedule[0], schedule[1], schedule[2], schedule[3], schedule[4], schedule[5]))
if args.monitor_frequency > 0:
    batch_per_epoch = x_train.shape[0] // args.batch_size
    cbs.append(callbacks.CollectActivationCallback(args.nb_epoch, args.monitor_frequency, args.batch_size, batch_per_epoch, ae, x_train[:5000], x_test[:5000], args.layers_to_monitor, "{}_activation_history".format(args.prefix)))

if not args.use_nat:
    if args.augmentation_ratio > 0:
        x_train_flow = data_object.get_train_flow(args.batch_size, args.augmentation_ratio)
        d = x_train_flow.next()
        ae.fit_generator(x_train_flow,
                         verbose=args.verbose,
                         steps_per_epoch=len(x_train) / args.batch_size,
                         epochs=args.nb_epoch,
                         callbacks = cbs,
                         validation_data=(x_test, x_test)
        )
    else:
        ae.fit(x_train, x_train,
               verbose=args.verbose,
               shuffle=True,
               epochs=args.nb_epoch,
               batch_size=args.batch_size,
               callbacks = cbs,
               validation_data=(x_test, x_test)
        )

else:
    import kohonen
    latent = sampler(x_train.shape[0], args.latent_dim)
    masterPermutation = np.arange(x_train.shape[0]).astype(np.int32)
    ae_with_nat = modelDict.ae_with_nat

    for epoch in range(args.nb_epoch):
        print "Epoch {}".format(epoch)
        if epoch % args.matching_frequency == 0:
            if args.distance_space == "latent":
                true_latent = encoder.predict(x_train, batch_size=args.batch_size)
                if epoch % args.frequency == 0:
                    vis.display_pair_distance_histogram(true_latent, latent[masterPermutation], args.prefix + "-pairdistance.png")
                    vis.display_pair_distance_histogram(true_latent, latent[masterPermutation], args.prefix + "-pairdistance-{}.png".format(epoch))
                true_latent_variances = np.var(true_latent, axis=0)
                print np.histogram(true_latent_variances)
                newPermutation = kohonen.batchPairing(latent, true_latent, masterPermutation, args.min_items_in_matching, args.greedy_matching)
            elif args.distance_space == "pixel":
                target_reconstruction = generator.predict(latent, batch_size=args.batch_size)
                newPermutation = kohonen.batchPairing(kohonen.biflatten(target_reconstruction), kohonen.biflatten(x_train), masterPermutation, args.min_items_in_matching, args.greedy_matching)
            else:
                assert False, "Unknown distance_space {}".format(args.distance_space)
            fixedPointRatio = float(np.sum(newPermutation  == masterPermutation)) / len(masterPermutation)
            masterPermutation = newPermutation
            print "FixedPointRatio: {}".format(fixedPointRatio)

        ae_with_nat.fit([x_train,latent[masterPermutation]], x_train,
                        verbose=args.verbose,
                        shuffle=True,
                        epochs=1,
                        batch_size=args.batch_size,
                        validation_data=([x_test,latent[:len(x_test)]], x_test))
        for cb in cbs:
            cb.on_epoch_end(epoch, 0)

load_models.saveModel(ae, args.prefix + "_model")
load_models.saveModel(encoder, args.prefix + "_encoder")
load_models.saveModel(encoder_var, args.prefix + "_encoder_var")
load_models.saveModel(generator, args.prefix + "_generator")
if args.decoder == "gaussian":
    load_models.saveModel(modelDict.generator_mixture, args.prefix + "_generator_mixture")

vis.displayGaussian(args, modelDict, x_train, args.prefix + "-dots")

# display randomly generated images
vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "%s-random" % args.prefix, batch_size=args.batch_size)


vis.displaySet(x_test[:args.batch_size], args.batch_size, args.batch_size, ae, "%s-test" % args.prefix)
vis.displaySet(x_train[:args.batch_size], args.batch_size, args.batch_size, ae, "%s-train" % args.prefix)

# display image interpolation
vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix, anchor_indices = data_object.anchor_indices)

vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))
