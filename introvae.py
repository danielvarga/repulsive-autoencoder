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
#data_path = '/home/ubuntu/celebA-HQ-256x256/'

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

reconst_latent_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='reconst_latent_input')
zr_mean, zr_log_var = encoder(decoder(reconst_latent_input))
zr_mean_ng, zr_log_var_ng = encoder(tf.stop_gradient(decoder(reconst_latent_input)))

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

#def size_loss(mean):
#    return K.mean(0.5 * K.sum(K.square(mean), axis=-1))

#def variance_loss(log_var):
#    return K.mean(0.5 * K.sum(-1 - log_var + K.exp(log_var), axis=-1))

#def reg_loss(mean, log_var):
#    return size_loss(mean) + variance_loss(log_var)

# sse or mse?
def mse_loss(x, x_decoded):
    original_dim = np.float32(np.prod(args.original_shape))
    return K.mean(original_dim * mean_squared_error(x, x_decoded))

def sse_loss(x, x_decoded):
    return 0.5 * K.sum(K.square(x_decoded - x))

#def reg_loss(mean, log_var):
#    return 0.5 * K.sum(- 1 - log_var + K.square(mean) + K.exp(log_var))

def reg_loss(mean, log_var):
    return  K.mean(0.5 * K.sum(- 1 - log_var + K.square(mean) + K.exp(log_var), axis=-1))

l_reg_z = reg_loss(z_mean, z_log_var)
l_reg_zr_ng = reg_loss(zr_mean_ng, zr_log_var_ng)
l_reg_zpp_ng = reg_loss(zpp_mean_ng, zpp_log_var_ng)

l_ae = mse_loss(encoder_input, xr)
#l_ae = sse_loss(encoder_input, xr)
ae_loss = args.beta * l_ae

encoder_l_adv = l_reg_z + args.alpha * K.maximum(0., args.m - l_reg_zr_ng) + args.alpha * K.maximum(0., args.m - l_reg_zpp_ng)
encoder_loss = encoder_l_adv + args.beta * l_ae

l_reg_zr = reg_loss(zr_mean, zr_log_var)
l_reg_zpp = reg_loss(zpp_mean, zpp_log_var)

decoder_l_adv = args.alpha * l_reg_zr + args.alpha * l_reg_zpp
decoder_loss = decoder_l_adv + args.beta * l_ae

print('Encoder')
encoder.summary()
print('Generator')
decoder.summary()
print('Start training')

iterations = x_generator.__len__()

encoder_params = encoder.trainable_weights
decoder_params = decoder.trainable_weights

print('# encoder params: ', len(encoder_params))
print('# decoder params: ', len(decoder_params))

print('Define train step operations')

if args.simple_update:
    encoder_grads = encoder_optimizer.compute_gradients(encoder_loss, var_list=encoder_params)
    decoder_grads = decoder_optimizer.compute_gradients(decoder_loss, var_list=decoder_params)
    encoder_apply_grads_op = encoder_optimizer.apply_gradients(encoder_grads)
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

# original method
# encoder_train_op = encoder_optimizer.minimize(encoder_loss, var_list=encoder_params)
# decoder_train_op = decoder_optimizer.minimize(decoder_loss, var_list=decoder_params)


for v in encoder_params:
    tf.summary.histogram(v.name, v)
for v in decoder_params:
    tf.summary.histogram(v.name, v)

summary_op = tf.summary.merge_all()

print('Start session')
global_iters = 0
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    summary_writer = tf.summary.FileWriter("./tflog/", graph=tf.get_default_graph())
    saver = tf.train.Saver()
    if args.modelPath is not None and tf.train.checkpoint_exists(args.modelPath):
        saver.restore(session, args.modelPath)
        print('Model restored from ' + args.modelPath)

    for epoch in range(args.nb_epoch):
        for iteration in range(iterations):
            global_iters += 1

            x = x_generator.__getitem__(iteration)
            z_p = np.random.normal(loc=0.0, scale=1.0, size=(args.batch_size, args.latent_dim))
            z_x, x_r, x_p = session.run([z, xr, decoder_output], feed_dict={encoder_input: x, decoder_input: z_p})

            _ = session.run([encoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
            _ = session.run([decoder_apply_grads_op], feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})

            if global_iters % 10 == 0:
                summary, = session.run([summary_op], feed_dict={encoder_input: x})
                summary_writer.add_summary(summary, global_iters)

                if args.modelPath is not None:
                    saver.save(session, args.modelPath, global_step=global_iters)
                    print('Saved model to ' + args.modelPath)

            if (global_iters % args.frequency) == 0:
                enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np, decoder_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np = \
                 session.run([encoder_loss, l_ae, l_reg_z, l_reg_zr_ng, l_reg_zpp_ng, decoder_loss, l_ae, l_reg_zr, l_reg_zpp],
                             feed_dict={encoder_input: x, reconst_latent_input: z_x, sampled_latent_input: z_p})
                print('Epoch: {}/{}, iteration: {}/{}'.format(epoch+1, args.nb_epoch, iteration+1, iterations))
                print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))
                print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}'.format(decoder_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np))

        n_x = 5
        n_y = args.batch_size // n_x
        print('Save original images.')
        vis.plotImages(x, n_x, n_y, "{}_original_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
        print('Save generated images.')
        vis.plotImages(x_p, n_x, n_y, "{}_sampled_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)
        print('Save reconstructed images.')
        vis.plotImages(x_r, n_x, n_y, "{}_reconstructed_epoch{}_iter{}".format(args.prefix, epoch + 1, global_iters), text=None)

        x_generator.on_epoch_end()
print('OK')
