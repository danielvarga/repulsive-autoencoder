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

data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
x_true_flow = data_object.get_train_flow(args.batch_size)
args.original_shape = x_train.shape[1:]


print('Build networks')

encoder_channels = model_resnet.default_channels('encoder', args.shape)
decoder_channels = model_resnet.default_channels('decoder', args.shape)

if args.shape == (64, 64):
    encoder_layers = model_resnet.encoder_layers_introvae_64(encoder_channels, args.encoder_use_bn)
    decoder_layers = model_resnet.decoder_layers_introvae_64(decoder_channels, args.decoder_use_bn)
elif args.shape == (1024, 1024):
    encoder_layers = model_resnet.encoder_layers_introvae_1024(encoder_channels, args.encoder_use_bn)
    decoder_layers = model_resnet.decoder_layers_introvae_1024(decoder_channels, args.decoder_use_bn)
else:
    assert False, 'unknown input size ' + args.shape

encoder_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='encoder_input')
decoder_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='decoder_input')

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
zr_mean, zr_log_var = encoder(xr)
zr_mean_ng, zr_log_var_ng = encoder(tf.stop_gradient(xr))

zp = K.random_normal(shape=(args.batch_size, args.latent_dim), mean=0.)

xp = decoder(zp)
zp_mean, zp_log_var = encoder(xp)
zp_mean_ng, zp_log_var_ng = encoder(tf.stop_gradient(xp))

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

def l_reg(mean, log_var):
    return  K.mean(0.5 * K.sum(-1 + log_var - K.square(mean) - K.exp(log_var), axis=-1))

def l_ae(x, x_decoded):
    original_dim = np.float32(np.prod(args.original_shape))
    return K.mean(original_dim * K.sum(mean_squared_error(x, x_decoded), axis=-1))


l_reg_z = l_reg(z_mean, z_log_var)
l_reg_zr_ng = l_reg(zr_mean_ng, zr_log_var_ng)
l_reg_zp_ng = l_reg(zp_mean_ng, zp_log_var_ng)
l_reconstruction = l_ae(encoder_input, xr)
encoder_loss = l_reg_z + args.alpha * K.maximum(0., args.m - l_reg_zr_ng) + args.alpha * K.maximum(0., args.m - l_reg_zp_ng) + args.beta * l_reconstruction

l_reg_zr = l_reg(zr_mean, zr_log_var)
l_reg_zp = l_reg(zp_mean, zp_log_var)
decoder_loss = args.alpha * l_reg_zr + args.alpha * l_reg_zp + args.beta * l_reconstruction

print('Encoder')
encoder.summary()
print('Generator')
decoder.summary()

print('Start training')

iterations = x_train.shape[0] // args.batch_size

encoder_params = encoder.trainable_weights
decoder_params = decoder.trainable_weights

print('Define train step operations')
encoder_train_op = encoder_optimizer.minimize(encoder_loss, var_list=encoder_params)
decoder_train_op = decoder_optimizer.minimize(decoder_loss, var_list=decoder_params)

print('Start session')
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    # summary_writer = tf.summary.FileWriter("./tmp/tf_logdir/", graph=tf.get_default_graph())
    for epoch in range(args.nb_epoch):
        for iteration in range(iterations):
            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch, args.nb_epoch, iteration, iterations))

            x_true = x_true_flow.next()[0]
            # z_true = session.run(z, feed_dict={encoder_input: x_true})
            # z_p = np.random.normal(size=(args.batch_size, args.latent_dim))
            # x_r = session.run(decoder_output, feed_dict={decoder_input: z_true})
            # x_p = session.run(decoder_output, feed_dict={decoder_input: z_p})

            _, enc_loss, ae_l, z_lreg, zr_ng_lreg, zp_ng_lreg = session.run([encoder_train_op, encoder_loss, l_reconstruction, l_reg_z, l_reg_zr_ng, l_reg_zp_ng],
                                                                            feed_dict={encoder_input: x_true})

            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zp_ng: {}'.format(enc_loss, ae_l, z_lreg, zr_ng_lreg, zp_ng_lreg))

            _, dec_loss, ae_l, zr_lreg, zp_lreg = session.run([decoder_train_op, decoder_loss, l_reconstruction, l_reg_zr, l_reg_zp],
                                                              feed_dict={encoder_input: x_true})

            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zp: {}'.format(dec_loss, ae_l, zr_lreg, zp_lreg))

            # print('Epoch: {}/{}, iteration: {}/{}, enc_loss: {}, dec_loss: {}'.format(epoch, args.nb_epoch, iteration, iterations, enc_loss, dec_loss))
print('OK')
