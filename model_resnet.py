import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add, \
    AveragePooling2D, Input, Dense, Flatten, UpSampling2D, Layer, Reshape, Concatenate, Lambda

from keras.models import Model

def default_channels(model_type, input_shape):
    if input_shape == (1024, 1024):
        encoder_channels = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        decoder_channels = [512, 512, 512, 512, 256, 128, 64, 32, 16, 16]
    elif input_shape == (64, 64):
        wideness = 1
        encoder_channels = [c*wideness for c in [16, 32, 64, 128, 256]]
        decoder_channels = [c*wideness for c in [256, 256, 128, 64, 32, 16, 16]]
    else:
        assert False, 'unknown input size ' + input_size

    if model_type == 'encoder':
        return encoder_channels
    elif model_type == 'decoder':
        return decoder_channels
    else:
        assert False, 'unknown model type'


class Encoder(object):
    pass


class Decoder(object):
    pass


class ResnetEncoder(Encoder):
    def __init__(self, args):
        self.args = args
        self.shape = args.shape
        self.latent_dim = args.latent_dim
        self.bn_allowed = args.encoder_use_bn
        self.channels = default_channels('encoder', self.shape)

    def __call__(self, x):
        if self.args.shape == (64, 64):
            layers = encoder_layers_introvae_64(self.channels, self.bn_allowed)
        elif self.args.shape == (1024, 1024):
            layers = encoder_layers_introvae_1024(self.channels, self.bn_allowed)
        h = x
        for layer in layers:
            h = layer(h)
        return h


class ResnetDecoder(Decoder):
    def __init__(self, args):
        self.args = args
        self.shape = args.shape
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.bn_allowed = args.decoder_use_bn
        self.channels = default_channels('decoder', self.shape)

    def __call__(self, recons_input):
        if self.args.shape == (64, 64):
            layers = decoder_layers_introvae_64(self.channels, self.bn_allowed)
        elif self.args.shape == (1024, 1024):
            layers = decoder_layers_introvae_1024(self.channels, self.bn_allowed)

        generator_input = Input(batch_shape=(self.batch_size, self.latent_dim))
        generator_output = generator_input
        for layer in layers:
            generator_output = layer(generator_output)

        generator_model = Model(generator_input, generator_output)
        recons_output = generator_model(recons_input)

        return generator_input, recons_output, generator_output


def resblock_and_avgpool_layers(kernels, filters, block, bn_allowed, last_activation="relu"):
    layers = []
    layers.append(residual_block('encoder',
                                 kernels=kernels,
                                 filters=filters,
                                 block=block,
                                 bn_allowed=bn_allowed, last_activation=last_activation))
    layers.append(AveragePooling2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='valid',
                                   name='avgpool'+ str(block)))
    return layers


def encoder_layers_introvae_1024(channels, bn_allowed):
    layers = []
    layers.append(Conv2D(channels[0], (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv' + str(0)))
    if bn_allowed:
        layers.append(BatchNormalization(axis=-1))
    layers.append(Activation('relu'))
    layers.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='avgpool' + str(1)))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[1], block=1, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[2], block=2, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[3], block=3, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[4], block=4, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[5], block=5, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[6], block=6, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(3, 3), (3, 3)], channels[7], block=7, bn_allowed=bn_allowed))
    layers.append(residual_block('encoder', kernels=[(3, 3), (3, 3)], filters=channels[8], block=8, bn_allowed=bn_allowed, last_activation="linear"))
    layers.append(Flatten())
    return layers

def encoder_layers_introvae_64(channels, bn_allowed):
    layers = []
    layers.append(Conv2D(channels[0], (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv' + str(0)))
    if bn_allowed:
        layers.append(BatchNormalization(axis=-1))
    layers.append(Activation('relu'))
    layers.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='avgpool' + str(0)))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[1], block=1, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[2], block=2, bn_allowed=bn_allowed))
    layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[3], block=3, bn_allowed=bn_allowed, last_activation="linear"))

    #layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[4], block=4, bn_allowed=bn_allowed))
    #layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[5], block=5, bn_allowed=bn_allowed))
    #layers.extend(resblock_and_avgpool_layers([(1, 1), (3, 3), (3, 3)], channels[6], block=6, bn_allowed=bn_allowed))
    #layers.extend(resblock_and_avgpool_layers([(3, 3), (3, 3)], channels[7], block=7, bn_allowed=bn_allowed))
    #layers.extend(residual_block('encoder', kernels=[(3, 3), (3, 3)], filters=channels[8], block=8, bn_allowed=bn_allowed))
    layers.append(Flatten())
    return layers


def upsample_and_resblock_layers(kernels, filters, block, bn_allowed):
    layers = []
    layers.append(UpSampling2D(size=(2, 2)))
    layers.append(residual_block('decoder', kernels=kernels, filters=filters, block=block, bn_allowed=bn_allowed))
    return layers


def decoder_layers_introvae_1024(channels, bn_allowed):
    layers = []
    layers.append(Dense(channels[0] * 4 * 4))
    layers.append(Activation('relu'))
    layers.append(Reshape((4, 4, channels[1])))
    layers.extend(residual_block('decoder', kernels=[(3, 3), (3, 3)], filters=channels[2], block=2, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[3], block=3, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[4], block=4, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[5], block=5, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[6], block=6, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[7], block=7, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[8], block=8, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[9], block=9, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[10], block=10, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[11], block=11, bn_allowed=bn_allowed))
    layers.append(Conv2D(3, (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv' + str(len(channels) + 1)))
    return layers

def decoder_layers_introvae_64(channels, bn_allowed):
    layers = []
    layers.append(Dense(channels[0] * 4 * 4))
    layers.append(Activation('relu'))
    layers.append(Reshape((4, 4, channels[1])))
    layers.append(residual_block('decoder', kernels=[(3, 3), (3, 3)], filters=channels[2], block=2, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[3], block=3, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[4], block=4, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[5], block=5, bn_allowed=bn_allowed))
    layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[6], block=6, bn_allowed=bn_allowed))
    #layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[7], block=7, bn_allowed=bn_allowed))
    #layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[8], block=8, bn_allowed=bn_allowed))
    #layers.extend(upsample_and_resblock_layers([(1, 1), (3, 3), (3, 3)], channels[9], block=9, bn_allowed=bn_allowed))
    #layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[10], block=10, bn_allowed=bn_allowed))
    #layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels[11], block=11, bn_allowed=bn_allowed))
    layers.append(Conv2D(3, (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv' + str(len(channels) + 1)))
    return layers


def residual_block(model_type, kernels, filters, block, bn_allowed, stage='a', last_activation="relu"):

    def identity_block(input_tensor, filters=filters):
        print(last_activation)
        if isinstance(filters, int):
            filters = [filters] * len(kernels)
        assert len(filters) == len(kernels), 'Number of filters and number of kernels differs.'
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        bn_name_base = model_type + '_bn_' + stage + str(block) + '_branch_'
        conv_name_base = model_type + 'conv_' + stage + str(block) + '_branch_'

        x = input_tensor

        for idx in range(len(filters)):
            x = Conv2D(filters[idx],
                       kernels[idx],
                       padding='same',
                       kernel_initializer='he_normal',
                       name=conv_name_base + str(idx))(x)
            if bn_allowed:
                x = BatchNormalization(axis=bn_axis,
                                       name=bn_name_base + str(idx))(x)
            if idx <= len(filters) - 1:
                x = Activation('relu')(x)

        if K.int_shape(input_tensor[-1]) != K.int_shape(x[-1]):
            if model_type == 'encoder':
                # print('resblock input tensor int_shape: ', K.int_shape(input_tensor))
                # print('resblock input tensor shape: ', K.shape(input_tensor))
                # print('input_shape: ', input_tensor.shape)
                padding = K.zeros(K.int_shape(input_tensor), name='padding')
                #print(padding._keras_history)
                #input_tensor = Concatenate()([input_tensor, padding])
                input_tensor = Lambda(lambda y: K.concatenate([y, padding]))(input_tensor)
            if model_type == 'decoder':
                input_tensor = Conv2D(K.int_shape(x)[-1], 1)(input_tensor)

        x = Add()([x, input_tensor])
        x = Activation(last_activation)(x)
        # print('Residual block output shape: ', K.int_shape(x))
        return x
    # return Lambda(lambda x: identity_block(x))
    return identity_block
