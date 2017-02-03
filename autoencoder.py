'''This AE differs from a standard VAE (https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py)
in the following ways:

- The latent variances are not learned, they are simply set to 0.
- A normalization step takes the latent variables to a sphere surface.
- The regularization loss is changed to an energy term that
  corresponds to a pairwise repulsive force between the encoded
  elements of the minibatch.
'''

import numpy as np
import params
import vis


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
(x_train, x_test) = data.load(args.dataset, args.trainSize, args.testSize, shape=args.shape, color=args.color)
args.original_shape = x_train.shape[1:]

import model
ae, encoder, encoder_var, generator, latent_critic, critic_layers  = model.build_model(args)
ae.summary()
latent_critic.summary()

sampler = model.sampler_factory(args, x_train)

import callbacks
cbs = [callbacks.FlushCallback()]
cbs.append(callbacks.get_lr_scheduler(args.nb_epoch, args.lr))
cbs.append(callbacks.ImageDisplayCallback(
    x_train, x_test, args,
    ae, encoder, encoder_var, generator, sampler))
for schedule in args.weight_schedules:
    if schedule[1] != schedule[2]:
        cbs.append(callbacks.WeightSchedulerCallback(args.nb_epoch, schedule[0], schedule[1], schedule[2], schedule[3], schedule[4], schedule[5]))


sampler = model.sampler_factory(args, x_train)


for epoch in range(args.nb_epoch):
    number_of_batches = int(x_train.shape[0]/args.batch_size)
    for ind in range(number_of_batches):

	fake_latent = [[0] * args.latent_dim] * args.batch_size


	for layer in critic_layers:
	    weights = layer.get_weights()
	    #print(weights)

	    weights[0] = np.clip(weights[0], -0.01, 0.01)
	    #weights[1] = np.clip(weights[1], -0.01, 0.01)
	    layer.set_weights(weights)
	
	    #print(weights)

	# Train the ae

	image_batch = x_train[ind*args.batch_size:(ind+1)*args.batch_size]
	#print(image_batch.shape)
	#ae_loss = ae.train_on_batch([np.array(image_batch), np.array(fake_latent)], image_batch)
	ae_loss = ae.fit([np.array(image_batch), np.array(fake_latent)],
		image_batch, batch_size=args.batch_size, nb_epoch=2)

	
	
	# Train the latent discriminator/critic
	ae.trainable = False

	true_sample = sampler(args.batch_size, args.latent_dim)
	images_encoded = encoder.predict([np.array(image_batch), np.array(fake_latent)], batch_size = args.batch_size)
	
	x = np.concatenate((true_sample, images_encoded))
	y = [1] * args.batch_size + [0] * args.batch_size
        #print(x.shape)
	image_batch = np.concatenate((image_batch, image_batch))
	#d_loss = latent_critic.train_on_batch(x, y)
	#y = latent_critic.predict(x, batch_size=args.batch_size)
	d_loss = latent_critic.fit([np.array(image_batch), np.array(x)], y, batch_size=args.batch_size, nb_epoch=3, verbose=1)
	#print(y)
	ae.trainable = True
	
	#print("batch end")
	#print("batch %d d_loss: %f" % (ind, d_loss))

	"""    
	
	discriminator.trainable = False
	noise = sampler(args.batch_size, latent_dim)
	g_loss = discriminator_on_generator.train_on_batch(noise, [1]*args.batch_size)
	discriminator.trainable = True
	"""
    print("Epoch ended.")
    vis.displaySet([np.array(x_train[:args.batch_size]),np.array(x)], args.batch_size, 100, ae, "%s-train" % args.prefix)

print("Train ended.")
	
"""

ae.fit(x_train, x_train,
       verbose=args.verbose,
       shuffle=True,
       nb_epoch=args.nb_epoch,
       batch_size=args.batch_size,
       callbacks = cbs,
       validation_data=(x_test, x_test))

"""

vis.saveModel(ae, args.prefix + "_model")
vis.saveModel(encoder, args.prefix + "_encoder")
vis.saveModel(encoder_var, args.prefix + "_encoder_var")
vis.saveModel(generator, args.prefix + "_generator")

vis.displayGaussian(args, ae, x_train, args.prefix + "-dots")

# display randomly generated images
vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "%s-random" % args.prefix, batch_size=args.batch_size)


vis.displaySet(x_test[:args.batch_size], args.batch_size, 100, ae, "%s-test" % args.prefix)
vis.displaySet(x_train[:args.batch_size], args.batch_size, 100, ae, "%s-train" % args.prefix)

# display image interpolation
if args.decoder != "gaussian":
    vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix)

vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))


# # display a 2D plot of the validation set in the latent space
# vis.latentScatter(encoder, x_test, batch_size, args.prefix+"-fig1")

# display 2D manifolds of images
# show_manifolds = False
# if show_manifolds:
#     for y in range(1, args.latent_dim-1):
#         vis.displayImageManifold(30, args.latent_dim, generator, height, width, 0, y, y+1, "%s-manifold%d" % (args.prefix, y), batch_size=batch_size)

