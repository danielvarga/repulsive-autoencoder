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

# data_object = data.load(args.dataset, shape=args.shape, color=args.color)
# (x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
# x_true_flow = data_object.get_train_flow(args.batch_size)
# args.original_shape = x_train.shape[1:]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, id_list, batch_size=32, shape=(64, 64), n_channels=3, shuffle=True):
        'Initialization'
        self.data_path = data_path
        self.shape = shape
        self.batch_size = batch_size
        self.id_list = id_list
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.id_list) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Find list of IDs
        id_list_temp = [self.id_list[i] for i in indexes]

        # Generate data
        X = self.__data_generation(id_list_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.id_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, id_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *shape, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.shape, self.n_channels))

        # Generate data
        for i, id_item in enumerate(id_list_temp):
            # Store sample
            X[i,] = np.load(self.data_path + id_item)
        X = X.astype('float32') / 255.
        return X



data_path = '/home/csadrian/download-celebA-HQ/256x256/'
#images = []
#for i in range(args.batch_size):
#    img = np.load(data_path + 'imgHQ' + str(i).zfill(5) + '.npy')
#    images.append(img)
#x_train_raw = np.concatenate(images, axis=0)
#args.original_shape = x_train_raw.shape[1:]
#print(args.original_shape)
#x_train = x_train_raw.astype('float32') / 255.
#print(x_train.shape)
#assert not np.any(np.isnan(x_train)), 'Input does not contain any nan.'
if args.color:
    args.n_channels = 3
else:
    args.n_channels = 1
args.original_shape = args.shape + (args.n_channels, )
print(args.original_shape)
id_list = [item for item in os.listdir(data_path) if item.endswith(".npy")]
if len(id_list) > args.trainSize:
    id_list_train = id_list[:args.trainSize]
else:
    id_list_train = id_list
print('Number of input images: ', len(id_list_train))
x_generator = DataGenerator(data_path, id_list_train, batch_size=args.batch_size, shape=args.shape, n_channels=args.n_channels)
#print('Dataset - iterations: ', x_generator.__len__())
#sys.exit('Intentionally terminated')

print('Build networks')

encoder_channels = model_resnet.default_channels('encoder', args.shape)
decoder_channels = model_resnet.default_channels('decoder', args.shape)

if args.shape == (64, 64):
    encoder_layers = model_resnet.encoder_layers_introvae_64(encoder_channels, args.encoder_use_bn)
    decoder_layers = model_resnet.decoder_layers_introvae_64(decoder_channels, args.decoder_use_bn)
elif args.shape == (256, 256):
    encoder_layers = model_resnet.encoder_layers_introvae_256(encoder_channels, args.encoder_use_bn)
    decoder_layers = model_resnet.decoder_layers_introvae_256(decoder_channels, args.decoder_use_bn)
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
#reconst_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='reconst_input')
#zr_mean, zr_log_var = encoder(reconst_input)
#zr_mean_ng, zr_log_var_ng = encoder(tf.stop_gradient(reconst_input))

zp = K.random_normal(shape=(args.batch_size, args.latent_dim), mean=0., stddev=1.0)
xp = decoder(zp)
zp_mean, zp_log_var = encoder(xp)
zp_mean_ng, zp_log_var_ng = encoder(tf.stop_gradient(xp))

#sampled_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='sampled_input')
#zp_mean, zp_log_var = encoder(sampled_input)
#zp_mean_ng, zp_log_var_ng = encoder(tf.stop_gradient(sampled_input))

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
    return  K.mean(- 0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1))

#def l_reg(mean, log_var):
#    return l_size(mean) + l_size(log_var)

def l_size(mean):
    return K.mean(0.5 * K.sum(K.abs(mean), axis=-1))

def l_variance(log_var):
    return K.mean(0.5 * K.sum(-1 - log_var + K.exp(log_var), axis=-1))

l_size_z = l_size(z_mean)
l_var_z = l_variance(z_log_var)

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

#iterations = x_train.shape[0] // args.batch_size
iterations = x_generator.__len__()

encoder_params = encoder.trainable_weights
decoder_params = decoder.trainable_weights

print('Define train step operations')
encoder_train_op = encoder_optimizer.minimize(encoder_loss, var_list=encoder_params)
decoder_train_op = decoder_optimizer.minimize(decoder_loss, var_list=decoder_params)

vae = Model(inputs=encoder_input, outputs=xr)
vae_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
vae_loss = l_reconstruction + l_reg_z
vae_train_op = vae_optimizer.minimize(vae_loss, var_list=vae.trainable_weights)

print('VAE')
vae.summary()

print('Start session')
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    # summary_writer = tf.summary.FileWriter("./tmp/tf_logdir/", graph=tf.get_default_graph())
    for epoch in range(args.nb_epoch):
        for iteration in range(iterations):
            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch, args.nb_epoch, iteration, iterations))

            x_true = x_generator.__getitem__(iteration)
            z_true = session.run(z, feed_dict={encoder_input: x_true})
            x_r = session.run(xr, feed_dict={encoder_input: x_true})
            #z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
            #x_p = session.run(decoder_output, feed_dict={decoder_input: z_p})
            x_p = session.run(xp, feed_dict={encoder_input: x_true})

            #_, enc_loss, ae_l, z_lreg, zr_ng_lreg, zp_ng_lreg = session.run([encoder_train_op, encoder_loss, l_reconstruction, l_reg_z, l_reg_zr_ng, l_reg_zp_ng],
                                                                            #feed_dict={encoder_input: x_true, reconst_input: x_r, sampled_input: x_p})
            #                                                                feed_dict={encoder_input: x_true})
            #print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zp_ng: {}'.format(enc_loss, ae_l, z_lreg, zr_ng_lreg, zp_ng_lreg))

            #_, dec_loss, ae_l, zr_lreg, zp_lreg = session.run([decoder_train_op, decoder_loss, l_reconstruction, l_reg_zr, l_reg_zp],
                                                              #feed_dict={encoder_input: x_true, reconst_input: x_r, sampled_input: x_p})
            #                                                  feed_dict={encoder_input: x_true})
            #print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zp: {}'.format(dec_loss, ae_l, zr_lreg, zp_lreg))

            _, vae_loss_value, l_rec_value, l_reg_value, l_size, l_var = session.run([vae_train_op, vae_loss, l_reconstruction, l_reg_z, l_size_z, l_var_z],
                                                                                     feed_dict={encoder_input: x_true})
            print(' VAE_loss: {}, l_rec:{}, l_reg: {}, l_size: {}, l_var: {}'.format(vae_loss_value, l_rec_value, l_reg_value, l_size, l_var))

        if (epoch + 1) % args.frequency == 0:
            n_x = 5
            n_y = args.batch_size // n_x
            print('Save original images.')
            vis.plotImages(x_true, n_x, n_y, "{}_original_epoch{}".format(args.prefix, epoch + 1), text=None)
            print('Save generated images.')
            vis.plotImages(x_p, n_x, n_y, "{}_sampled_epoch{}".format(args.prefix, epoch + 1), text=None)
            print('Save reconstructed images.')
            vis.plotImages(x_r, n_x, n_y, "{}_reconstructed_epoch{}".format(args.prefix, epoch + 1), text=None)
        x_generator.on_epoch_end()
print('OK')
