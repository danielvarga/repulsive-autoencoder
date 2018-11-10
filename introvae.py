import numpy as np
from keras.objectives import mean_squared_error
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD

import keras.backend as K

import params
import data
import load_models

import callbacks
import samplers

import model_resnet
from model import add_sampling
import tensorflow as tf
import sys
import os
import vis

args = params.getArgs()
print(args)

# set random seed
np.random.seed(10)

# limit memory usage
import keras

print('Keras version: ', keras.__version__)
if keras.backend._BACKEND == 'tensorflow':
    import tensorflow as tf

    print('Tensorflow version: ', tf.__version__)
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))

epsilon = 0.00001

print('Load data')

iterations = args.nb_epoch * args.trainSize // args.batch_size

###

K.set_image_data_format('channels_first')

def read_npy_file(item):
    data = np.transpose(np.load(item.decode()), (0,3,1,2))[0,:,:,:]
    return data.astype(np.float32)

data_path = '/mnt/g2big/datasets/celeba/celebA-HQ-256x256/'
#data_path = '/home/ubuntu/celebA-HQ-256x256/'

iterations_per_epoch = args.trainSize // args.batch_size

def create_dataset(path, batch_size, limit):
    dataset = tf.data.Dataset.list_files(path, shuffle=True) \
        .take((limit // batch_size) * batch_size) \
        .map(lambda x: tf.py_func(read_npy_file, [x], [tf.float32])) \
        .map(lambda x: x / 255.) \
        .batch(args.batch_size) \
        .repeat() \
        .prefetch(2)
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    get_next = iterator.get_next()
    return (dataset, iterator, iterator_init_op, get_next)

train_dataset, train_iterator, train_iterator_init_op, train_next \
     = create_dataset(data_path + "train/*.npy", args.batch_size, args.trainSize)

test_dataset, test_iterator, test_iterator_init_op, test_next \
     = create_dataset(data_path + "test/*.npy", args.batch_size, args.testSize)

fixed_dataset, fixed_iterator, fixed_iterator_init_op, fixed_next \
     = create_dataset(data_path + "test/*.npy", args.batch_size, args.testSize)

###

if args.color:
    args.n_channels = 3
else:
    args.n_channels = 1
#args.original_shape = args.shape + (args.n_channels, )
args.original_shape = (args.n_channels, ) + args.shape
print(args.original_shape)

print('Build networks')

encoder_layers = model_resnet.encoder_layers_introvae(args.shape, args.base_filter_num, args.encoder_use_bn)
decoder_layers = model_resnet.decoder_layers_introvae(args.shape, args.base_filter_num, args.decoder_use_bn)

assert(args.shape in [(128,128), (256, 256), (512, 512), (1024, 1024)], "Model not available for input shape.")

encoder_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='encoder_input')
decoder_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='decoder_input')

#encoder_input = tf.placeholder(tf.float32, shape=[args.batch_size] + list(args.original_shape), name='encoder_input')
#decoder_input = tf.placeholder(tf.float32, shape=(args.batch_size, args.latent_dim), name='encoder_input')

encoder_output = encoder_input
for layer in encoder_layers:
    encoder_output = layer(encoder_output)

decoder_output = decoder_input
for layer in decoder_layers:
    decoder_output = layer(decoder_output)

z, z_mean, z_log_var = add_sampling(encoder_output, args.sampling, args.sampling_std, args.batch_size, args.latent_dim, args.encoder_wd)

encoder = Model(inputs=encoder_input, outputs=[z_mean, z_log_var])
decoder = Model(inputs=decoder_input, outputs=decoder_output)

xr = decoder(z)

reconst_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='reconst_latent_input')
zr_mean, zr_log_var = encoder(decoder(reconst_latent_input))
zr_mean_ng, zr_log_var_ng = encoder(tf.stop_gradient(decoder(reconst_latent_input)))
xr_lat = decoder(reconst_latent_input)

sampled_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='sampled_latent_input')
zpp_mean, zpp_log_var = encoder(decoder(sampled_latent_input))
zpp_mean_ng, zpp_log_var_ng = encoder(tf.stop_gradient(decoder(sampled_latent_input)))

print('Define optimizer')

if args.optimizer == 'adam':
    encoder_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    decoder_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
elif args.optimizer == 'rmsprop':
    encoder_optimizer = tf.train.RMSpropOptimizer(learning_rate=args.lr)
    decoder_optimizer = tf.train.RMSpropOptimizer(learning_rate=args.lr)
elif args.optimizer == 'sgd':
    encoder_optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
    decoder_optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)

print('Define loss functions')

def mse_loss(x, x_decoded):
    original_dim = np.float32(np.prod(args.original_shape))
    return K.mean(original_dim * mean_squared_error(x, x_decoded))

def reg_loss(mean, log_var):
    return K.mean(0.5 * K.sum(- 1 - log_var + K.square(mean) + K.exp(log_var), axis=-1))

def augmented_variance_loss(mean, log_var):
    variance = K.exp(z_log_var)
    # TODO Are you really-really sure it's not axis=1?
    mean_variance = K.var(mean, axis=0, keepdims=True)
    total_variance = variance + mean_variance
    loss = 0.5 * K.sum(-1 - K.log(total_variance) + total_variance, axis=-1)
    return K.mean(loss)

def size_loss(mean):
    loss = 0.5 * K.sum(K.square(mean), axis=-1)
    return K.mean(loss)

def reg_loss_new(mean, log_var):
    return augmented_variance_loss(mean, log_var) + size_loss(mean)

if args.newkl:
   reg_loss = reg_loss_new
   print("using newkl")


l_reg_z = reg_loss(z_mean, z_log_var)
l_reg_zr_ng = reg_loss(zr_mean_ng, zr_log_var_ng)
l_reg_zpp_ng = reg_loss(zpp_mean_ng, zpp_log_var_ng)

l_ae = mse_loss(encoder_input, xr)
l_ae2 = mse_loss(encoder_input, xr_lat)
ae_loss = args.beta * l_ae

encoder_l_adv = l_reg_z + args.alpha * K.maximum(0., args.m - l_reg_zr_ng) + args.alpha * K.maximum(0., args.m - l_reg_zpp_ng)
encoder_loss = encoder_l_adv + args.beta * l_ae

l_reg_zr = reg_loss(zr_mean, zr_log_var)
l_reg_zpp = reg_loss(zpp_mean, zpp_log_var)

decoder_l_adv = args.alpha * l_reg_zr + args.alpha * l_reg_zpp
decoder_loss = decoder_l_adv + args.beta * l_ae2

print('Encoder')
encoder.summary()
print('Generator')
decoder.summary()
print('Start training')

encoder_params = encoder.trainable_weights
decoder_params = decoder.trainable_weights

print('# encoder params: ', len(encoder_params))
print('# decoder params: ', len(decoder_params))

print('Define train step operations')

if args.simple_update:
    encoder_grads = encoder_optimizer.compute_gradients(encoder_loss, var_list=encoder_params)
    encoder_apply_grads_op = encoder_optimizer.apply_gradients(encoder_grads)

    decoder_grads = decoder_optimizer.compute_gradients(decoder_loss, var_list=decoder_params)
    decoder_apply_grads_op = decoder_optimizer.apply_gradients(decoder_grads)
else:
    encoder_ae_grads = tf.gradients(ae_loss, encoder_params)
    decoder_ae_grads = tf.gradients(ae_loss, decoder_params)
    encoder_adv_grads = tf.gradients(encoder_l_adv, encoder_params)
    encoder_grads = [x + y for x, y in zip(encoder_ae_grads, encoder_adv_grads)]
    with tf.control_dependencies(decoder_ae_grads):
        encoder_apply_grads_op = encoder_optimizer.apply_gradients(zip(encoder_grads, encoder_params))
        with tf.control_dependencies([encoder_apply_grads_op]):
            decoder_adv_grads = tf.gradients(decoder_l_adv, decoder_params)
            decoder_grads = [x + y for x, y in zip(decoder_ae_grads, decoder_adv_grads)]
            decoder_apply_grads_op = decoder_optimizer.apply_gradients(zip(decoder_grads, decoder_params))


for v in encoder_params:
    tf.summary.histogram(v.name, v)
for v in decoder_params:
    tf.summary.histogram(v.name, v)

summary_op = tf.summary.merge_all()
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
print('Start session')
global_iters = 0
start_epoch = 0
import time

with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run([init, train_iterator_init_op, test_iterator_init_op, fixed_iterator_init_op], options=run_opts)

    summary_writer = tf.summary.FileWriter(args.prefix+"/", graph=tf.get_default_graph())
    saver = tf.train.Saver()
    if args.modelPath is not None and tf.train.checkpoint_exists(args.modelPath):
        saver.restore(session, tf.train.latest_checkpoint(args.modelPath))
        print('Model restored from ' + args.modelPath)
        ckpt = tf.train.get_checkpoint_state(args.modelPath)
        global_iters = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        start_epoch = (global_iters * args.batch_size) // args.trainSize
    print('Global iters: ', global_iters)

    for iteration in range(iterations):

        epoch = global_iters * args.batch_size // args.trainSize
        global_iters += 1

        start_time = time.time()

        x = session.run(train_next)
        z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
        z_x, x_r, x_p = session.run([z, xr, decoder_output], feed_dict={encoder_input: x, decoder_input: z_p})

        _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p}, options=run_opts)
        _ = session.run([decoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p}, options=run_opts)

        print("--- %s seconds ---" % (time.time() - start_time))

        if global_iters % 10 == 0:
            summary, = session.run([summary_op], feed_dict={encoder_input: x})
            summary_writer.add_summary(summary, global_iters)

        if (global_iters % args.frequency) == 0:
            enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, decoder_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np = \
             session.run([encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, decoder_loss, l_ae, l_reg_zr, l_reg_zpp],
                         feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args.nb_epoch, iteration+1, iterations))
            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))
            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}'.format(decoder_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np))

        if ((global_iters % iterations_per_epoch == 0) and args.save_latent):
            test_latent_cloud, test_latent_cloud_log_var = [], []
            for i in range(10000 // args.batch_size):
                x_test = session.run(test_next)
                latent_cloud, latent_cloud_log_var = session.run([z_mean, z_log_var], feed_dict={encoder_input: x_test})
                test_latent_cloud.append(latent_cloud)
                test_latent_cloud_log_var.append(latent_cloud_log_var)
            filename = "{}_test_latent_mean_epoch{}_iter{}.npy".format(args.prefix, epoch+1, global_iters)
            print("Saving test latent pointcloud mean to {}".format(filename))
            np.save(filename, np.concatenate(test_latent_cloud, axis=0))
            filename = "{}_test_latent_log_var_epoch{}_iter{}.npy".format(args.prefix, epoch+1, global_iters)
            print("Saving test latent pointcloud log variance to {}".format(filename))
            np.save(filename, np.concatenate(test_latent_cloud_log_var, axis=0))


            big_latent_cloud, big_latent_cloud_log_var = [], []
            for i in range(args.latent_cloud_size // args.batch_size):
                x_latentcloud = session.run(fixed_next)
                latent_cloud, latent_cloud_log_var = session.run([z_mean, z_log_var], feed_dict={encoder_input: x_latentcloud})
                big_latent_cloud.append(latent_cloud)
                big_latent_cloud_log_var.append(latent_cloud_log_var)
            filename = "{}_big_latent_mean_epoch{}_iter{}.npy".format(args.prefix, epoch+1, global_iters)
            print("Saving big latent pointcloud mean to {}".format(filename))
            np.save(filename, np.concatenate(big_latent_cloud, axis=0))
            filename = "{}_big_latent_log_var_epoch{}_iter{}.npy".format(args.prefix, epoch+1, global_iters)
            print("Saving big latent pointcloud log variance to {}".format(filename))
            np.save(filename, np.concatenate(big_latent_cloud_log_var, axis=0))

            n_x = 5
            n_y = args.batch_size // n_x
            print('Save original images.')
            vis.plotImages(np.transpose(x, (0, 2, 3, 1))), n_x, n_y, "{}_original_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save generated images.')
            vis.plotImages(np.transpose(x_p, (0, 2, 3, 1)), n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
            print('Save reconstructed images.')
            vis.plotImages(np.transpose(x_r, (0, 2, 3, 1)), n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)

        if (epoch + 1) % 10 == 0:
            if args.modelPath is not None:
                saver.save(session, args.modelPath + "/model", global_step=global_iters)
                print('Saved model to ' + args.modelPath + "/model")



print('OK')
