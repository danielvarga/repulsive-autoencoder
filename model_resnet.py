import math
import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add, \
    AveragePooling2D, Input, Dense, Flatten, UpSampling2D, Layer, Reshape, Concatenate, Lambda

from keras.models import Model

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
        self.resnet_wideness = args.resnet_wideness

    def __call__(self, x):
        layers = encoder_layers_introvae(self.args.shape, self.channels * self.resnet_wideness, self.bn_allowed)
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
        self.resnet_wideness = args.resnet_wideness

    def __call__(self, recons_input):
        layers = decoder_layers_introvae(self.args.shape, self.channels * self.resnet_wideness, self.bn_allowed)

        generator_input = Input(batch_shape=(self.batch_size, self.latent_dim))
        generator_output = generator_input
        for layer in layers:
            generator_output = layer(generator_output)

        generator_model = Model(generator_input, generator_output)
        recons_output = generator_model(recons_input)

        return generator_input, recons_output, generator_output


def resblock_and_avgpool_layers(kernels, filters, block, bn_allowed, last_activation="relu"):
    layers = []
    layers.append(residual_block('encoder', kernels=kernels, filters=filters, block=block, bn_allowed=bn_allowed, last_activation=last_activation))
    layers.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='encoder_avgpool_'+ str(block)))
    return layers

def upsample_and_resblock_layers(kernels, filters, block, bn_allowed):
    layers = []
    layers.append(UpSampling2D(size=(2, 2), name='decoder_upsample_' + str(block)))
    layers.append(residual_block('decoder', kernels=kernels, filters=filters, block=block, bn_allowed=bn_allowed))
    return layers


def encoder_layers_introvae(image_size, base_channels, bn_allowed):

    layers = []
    layers.append(Conv2D(base_channels, (5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='encoder_conv_0'))
    if bn_allowed:
        layers.append(BatchNormalization(axis=1, name='encoder_bn_0'))
    layers.append(Activation('relu'))
    layers.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', name='encoder_avgpool_0'))

    map_size = image_size[0] // 2

    block = 1
    channels = base_channels * 2
    while map_size > 4:
        layers.extend(resblock_and_avgpool_layers([(3, 3), (3, 3)], channels, block=block, bn_allowed=bn_allowed))
        block += 1
        map_size = map_size // 2
        channels = channels * 2 if channels <= 256  else 512

    layers.append(residual_block('encoder', kernels=[(3, 3), (3, 3)], filters=channels, block=block, bn_allowed=bn_allowed, last_activation="linear"))
    layers.append(Flatten(name='encoder_reshape'))
    return layers



def decoder_layers_introvae(image_size, base_channels, bn_allowed):

    layers = []
    layers.append(Dense(512 * 4 * 4, name='decoder_dense'))
    layers.append(Activation('relu'))
    layers.append(Reshape((512, 4, 4), name='decoder_reshape'))
    layers.append(residual_block('decoder', kernels=[(3, 3), (3, 3)], filters=512, block=1, bn_allowed=bn_allowed))

    map_size = 4
    upsamples = int(math.log2(image_size[0]) - 2)
    block = 2
    channels = 512

    for i in range(upsamples - 6):
        layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], 512, block=block, bn_allowed=bn_allowed))
        map_size = map_size * 2
        block += 1

    while map_size < image_size[0]: # 4
        channels = channels // 2 if channels >= 32 else 16
        layers.extend(upsample_and_resblock_layers([(3, 3), (3, 3)], channels, block=block, bn_allowed=bn_allowed))
        map_size = map_size * 2
        block += 1

    layers.append(Conv2D(3, (5, 5), padding='same', kernel_initializer='he_normal', name='decoder_conv_0'))
    layers.append(Activation('sigmoid'))
    return layers

def residual_block(model_type, kernels, filters, block, bn_allowed, stage='a', last_activation="relu"):

    def identity_block(input_tensor, filters=filters):
        if isinstance(filters, int):
            filters = [filters] * len(kernels)
        assert len(filters) == len(kernels), 'Number of filters and number of kernels differs.'

        bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
        bn_name_base = model_type + '_resblock_bn_' + stage + str(block) + '_branch_'
        conv_name_base = model_type + '_resblock_conv_' + stage + str(block) + '_branch_'

        if K.int_shape(input_tensor[-1]) != filters[0]:
            input_tensor = Conv2D(filters[0], (1, 1), padding='same', kernel_initializer='glorot_normal', name=conv_name_base + str('00'), data_format='channels_first')(input_tensor)
            if bn_allowed:
                input_tensor = BatchNormalization(axis=bn_axis, name=bn_name_base + str('00'))(input_tensor)
            input_tensor = Activation('relu')(input_tensor)

        x = input_tensor
        for idx in range(len(filters)):
            x = Conv2D(filters[idx], kernels[idx], padding='same', kernel_initializer='he_normal', name=conv_name_base + str(idx), data_format='channels_first')(x)
            if bn_allowed:
                x = BatchNormalization(axis=bn_axis, name=bn_name_base + str(idx))(x)
            if idx <= len(filters) - 1:
                x = Activation('relu')(x)

        x = Add(name=model_type + '_resblock_add_' + stage + str(block))([x, input_tensor])
        x = Activation(last_activation)(x)
        # print('Residual block output shape: ', K.int_shape(x))
        return x
    # return Lambda(lambda x: identity_block(x))
    return identity_block
