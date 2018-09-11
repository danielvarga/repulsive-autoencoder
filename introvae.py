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

import model_introvae

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

print('Define loss functions')


def encoder_loss(x, xr):
    kl_loss_z = 0.5 * K.sum(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar),
                            axis=-1)
    kl_loss_zr = 0.5 * K.sum(1 + zr_logvar_ng - K.square(zr_mean_ng) - K.exp(zr_logvar_ng),
                             axis=-1)
    kl_loss_zp = 0.5 * K.sum(1 + zp_logvar_ng - K.square(zp_mean_ng) - K.exp(zp_logvar),
                             axis=-1)
    recon_loss = 0.5 * K.sum(mean_squared_error(x, xr),
                             axis=-1)
    loss = kl_loss_z + args.alpha * K.max(0, args.m - kl_loss_zr) + args.alpha * K.max(0, args.m - kl_loss_zp) + args.beta * recon_loss
    return loss


def decoder_loss(x, xr):
    kl_loss_zr = 0.5 * K.sum(1 + zr_logvar - K.square(zr_mean) - K.exp(zr_logvar),
                             axis=-1)
    kl_loss_zp = 0.5 * K.sum(1 + zp_logvar - K.square(zp_mean) - K.exp(zp_logvar),
                             axis=-1)
    recon_loss = 0.5 * K.sum(mean_squared_error(x, xr),
                             axis=-1)
    loss = args.alpha * kl_loss_zr + args.alpha * kl_loss_zp + args.beta * recon_loss
    return loss


print('Build networks')

encoder_channels = model_introvae.default_channels('encoder', args.shape)
decoder_channels = model_introvae.default_channels('decoder', args.shape)



print('decoder channels: ', decoder_channels)

encoder_input = Input(batch_shape=[args.batch_size] + list(args.original_shape), name='encoder_input')
decoder_input = Input(batch_shape=(args.batch_size, args.latent_dim), name='decoder_input')

print('Define optimizer')

if args.optimizer == 'adam':
    encoder_optimizer = Adam(lr=args.lr)
    decoder_optimizer = Adam(lr=args.lr)
elif args.optimizer == 'rmsprop':
    encoder_optimizer = RMSprop(lr=args.lr)
    decoder_optimizer = RMSprop(lr=args.lr)
elif args.optimizer == 'sgd':
    encoder_optimizer = SGD(lr=args.lr)
    decoder_optimizer = SGD(lr=args.lr)

sampler = samplers.gaussian_sampler

def build_networks(encoder_input, decoder_input):

    encoder_output = model_introvae.encoder_layers_introvae(encoder_input,
                                                            encoder_channels,
                                                            args.encoder_use_bn)
    decoder_output = model_introvae.decoder_layers_introvae(decoder_input,
                                                            decoder_channels,
                                                            args.decoder_use_bn)

    encoder = Model(inputs=encoder_input, outputs=encoder_output)
    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    print('Compile models.')
    encoder.compile(optimizer=encoder_optimizer, loss=encoder_loss)
    decoder.compile(optimizer=decoder_optimizer, loss=decoder_loss)

    return (encoder, decoder)

(encoder, decoder) = build_networks(encoder_input, decoder_input)



print('Encoder')
encoder.summary()
print('Generator')
decoder.summary()

print('OK')
