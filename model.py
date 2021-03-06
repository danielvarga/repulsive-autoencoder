'''
Standard VAE taken from https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''
import numpy as np

import dense
import model_conv_discgen
import model_gaussian
import model_strictly_gaussian
import model_resnet
import model_dcgan
import model_ladder
import samplers

from keras.layers import Input, Dense, Lambda, Reshape, Flatten, Activation
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import *
from keras.regularizers import l2

import activations
from loss import loss_factory
from arm import ArmLayer
from exp import AttrDict

def build_model(args):

    # INPUT
    x = Input(batch_shape=([args.batch_size] + list(args.original_shape)))

    # --------------------------------------------------------------------
    # ENCODER

    if args.encoder == "dense":

        encoder = dense.DenseEncoder(
            args.intermediate_dims,
            args.activation,
            args.encoder_wd,
            args.encoder_use_bn
        )

    elif args.encoder == "conv":

        encoder = model_conv_discgen.ConvEncoder(
            depth = args.depth,
            latent_dim = args.latent_dim,
            intermediate_dims = args.intermediate_dims,
            image_dims = args.original_shape,
            batch_size = args.batch_size,
            base_filter_num = args.base_filter_num
        )

    elif args.encoder == "dcgan":

        encoder = model_dcgan.DcganEncoder(args)

    elif args.encoder == "ladderDense":

        encoder = model_ladder.LadderDenseEncoder(args)

    elif args.encoder == "resnet":
        encoder = model_resnet.ResnetEncoder(args)


    hidden = encoder(x)

    # --------------------------------------------------------------------
    # LATENT

    if args.encoder == "ladderDense":
        z, z_mean, z_log_var = encoder.get_latent_code()
    else:
        z, z_mean, z_log_var = add_sampling(
            hidden,
            args.sampling,
            args.sampling_std,
            args.batch_size,
            args.latent_dim,
            args.encoder_wd
        )



    z_normed = Lambda(lambda z_unnormed: K.l2_normalize(z_unnormed, axis=-1))(z)
    if args.spherical:
        z = z_normed

    z_projected = Lambda(lambda z:
        K.reshape(
            K.dot(
                z,
                K.l2_normalize(K.random_normal_variable((args.latent_dim, 1), 0, 1), axis=-1)
            ),
            (args.batch_size,)
        )
    )(z)

    if args.decoder == "dense":
        decoder = dense.DenseDecoder(
            args.latent_dim,
            args.intermediate_dims,
            args.original_shape,
            args.activation,
            args.decoder_wd,
            args.decoder_use_bn
        )
    elif args.decoder == "conv":
        decoder = model_conv_discgen.ConvDecoder(
            depth = args.depth,
            latent_dim = args.latent_dim,
            intermediate_dims =args.intermediate_dims,
            image_dims = args.original_shape,
            batch_size = args.batch_size,
            base_filter_num = args.base_filter_num,
            wd = args.decoder_wd,
            use_bn = args.decoder_use_bn)
    elif args.decoder == "gaussian":
        decoder = model_gaussian.GaussianDecoder(args)
    elif args.decoder == "strictlyGaussian":
        decoder = model_strictly_gaussian.StrictlyGaussianDecoder(args, x)
    elif args.decoder == "resnet":
        decoder = model_resnet.ResnetDecoder(args)
    elif args.decoder =="dcgan":
        decoder = model_dcgan.DcganDecoder(args)
    elif args.decoder == "ladderDense":
        decoder = model_ladder.LadderDenseDecoder(args)
    elif args.decoder == "resnet":
        decoder = model_resnet.ResnetDecoder(args)

    decoder_fun_output = decoder(z)
    generator_input, recons_output, generator_output = decoder_fun_output[:3]

    encoder = Model(x, z_mean)
    encoder_var = Model(x, z_log_var)
    ae = Model(x, recons_output)
    generator = Model(generator_input, generator_output)

#    armLayer = ArmLayer(dict_size=1000, iteration=5, threshold=0.01, reconsCoef=1)
#    sparse_input = Flatten()(x)
#    sparse_input = armLayer(sparse_input)
#    sparse_output = Flatten()(recons_output)
#    sparse_output = armLayer(sparse_output)
    loss_features = AttrDict({
        "z_sampled": z,
        "z_mean": z_mean,
        "z_log_var": z_log_var,
        "z_normed": z_normed,
#        "sparse_input": sparse_input,
#        "sparse_output": sparse_output,
        "z_projected": z_projected
    })
    if args.use_nat:
        nat_input = Input(batch_shape=(args.batch_size, args.latent_dim), name="nat_input")
        ae_with_nat = Model([x, nat_input], recons_output)
        loss_features.nat_input = nat_input

    loss, metrics = loss_factory(ae, encoder, loss_features, args)


    if args.optimizer == "rmsprop":
        optimizer = RMSprop(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr = args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer

    ae.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    modelDict = AttrDict({})
    modelDict.ae = ae
    modelDict.encoder = encoder
    modelDict.encoder_var = encoder_var
    modelDict.generator = generator
    if args.use_nat:
        ae_with_nat.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        modelDict.ae_with_nat = ae_with_nat
    if args.decoder == "gaussian":
        generator_mixture_output = decoder_fun_output[3]
        generator_mixture = Model(generator_input, generator_mixture_output)
        modelDict.generator_mixture = generator_mixture

    return modelDict


def add_sampling(hidden, sampling, sampling_std, batch_size, latent_dim, wd):
    z_mean = Dense(latent_dim, kernel_regularizer=l2(wd))(hidden)
    if not sampling:
        z_log_var = Lambda(lambda x: 0*x, output_shape=[latent_dim])(z_mean)
        return z_mean, z_mean, z_log_var
    else:
        if sampling_std > 0:
            z_log_var = Lambda(lambda x: 0*x + K.log(K.square(sampling_std)), output_shape=[latent_dim])(z_mean)
        else:
            z_log_var = Dense(latent_dim, kernel_regularizer=l2(wd))(hidden)

        def norm(z):
            z_odd = z[:, 1::2]
            z_even = z[:, 0::2]
            z_divider = K.repeat_elements(K.sqrt(z_odd**2 + z_even**2), 2, 1)
            
            return z / z_divider

        # Uncomment if you want to place the mean on a toroid
        # z_mean = Lambda(norm)(z_mean)

        def sampling(inputs):
            z_mean, z_log_var = inputs
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        return z, z_mean, z_log_var

