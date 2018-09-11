import keras.backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, Add, \
    AveragePooling2D, Dense, Flatten, UpSampling2D, Layer, Reshape, Concatenate


def default_channels(model_type, input_shape):
    if input_shape == (64, 64):
        encoder_channels = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        decoder_channels = list(reversed(encoder_channels))
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


class IntrovaeEncoder(Encoder):
    def __init__(self, args):
        self.args = args
        self.shape = args.shape
        self.latent_dim = args.latent_dim
        self.bn_allowed = args.encoder_use_bn
        self.channels = default_channels('encoder',
                                         self.shape,
                                         self.latent_dim)

    def __call__(self, x):
        pass


class IntrovaeDecoder(Decoder):
    def __init__(self, args):
        self.args = args
        self.channels = default_channels('decoder',
                                         self.input_shape,
                                         self.latent_dim)

    def __call__(self, recons_input):
        pass


def resblock_and_avgpool_layers(input_tensor, kernels, num_filters, block, bn_allowed):
    x = input_tensor
    x = residual_block(x,
                       'encoder',
                       kernels=kernels,
                       filters=[num_filters] * len(kernels),
                       block=str(block),
                       bn_allowed=bn_allowed)
#    x = AveragePooling2D(pool_size=(2, 2),
#                         strides=None,
#                         padding='valid',
#                         name='avgpool'+ str(block))(x)
    return x


def encoder_layers_introvae(input_tensor, channels, bn_allowed):
    x = input_tensor
    for idx, channel in enumerate(channels):
        if idx == 0:
            x = Conv2D(channel,
                       (5, 5),
                       strides=(1, 1),
                       padding='same',
                       kernel_initializer='he_normal',
                       name='conv' + str(idx))(x)
            if bn_allowed:
                x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=(2, 2),
                                 strides=None,
                                 padding='valid',
                                 name='avgpool' + str(idx))(x)
        elif idx <= len(channels) - 3:
            x = resblock_and_avgpool_layers(x,
                                            [(1, 1), (3, 3), (3, 3)],
                                            channel,
                                            idx,
                                            bn_allowed)
        elif  idx == len(channels) - 2:
            x = resblock_and_avgpool_layers(x,
                                            [(3, 3), (3, 3)],
                                            channel,
                                            idx,
                                            bn_allowed)
        elif idx == len(channels) - 1:
            x = residual_block(x,
                               'encoder',
                               kernels=[(3, 3), (3, 3)],
                               filters=[channel] * 2,
                               block=str(idx),
                               bn_allowed=bn_allowed)
    x = Flatten()(x)
    return x


def upsample_and_resblock_layers(input_tensor, kernels, num_filters, block, bn_allowed):
    x = input_tensor
    x = UpSampling2D(size=(2, 2))(x)
    x = residual_block(x,
                       'decoder',
                       kernels=kernels,
                       filters=[num_filters] * len(kernels),
                       block=str(block),
                       bn_allowed=bn_allowed)
    return x


def decoder_layers_introvae(input_tensor, channels, bn_allowed):
    # print('type of channels: ', type(channels), ', channels: ', channels)
    x = input_tensor
    for idx, channel in enumerate(channels):
        # print('idx: {}, channel: {}'.format(idx, channel))
        if idx == 0:
            x = Dense(channel * 4 * 4)(x)
            x = Activation('relu')(x)
        elif idx == 1:
            x = Reshape((4, 4, channel))(x)
            x = residual_block(x, 
                               'decoder',
                                kernels=[(3, 3), (3, 3)],
                                filters=[channel] * 2,
                                block=str(idx),
                                bn_allowed=bn_allowed)
        elif idx < 4:
            x = upsample_and_resblock_layers(x,
                                             [(3, 3), (3, 3)],
                                             channel,
                                             idx,
                                             bn_allowed)
        else:
            x = upsample_and_resblock_layers(x,
                                             [(1, 1), (3, 3), (3, 3)],
                                             channel,
                                             idx,
                                             bn_allowed)
    x = upsample_and_resblock_layers(x,
                                     [(3, 3), (3, 3)],
                                     16,
                                     len(channels),
                                     bn_allowed)
    x = Conv2D(3,
               (5, 5),
               strides=(1, 1),
               padding='same',
               kernel_initializer='he_normal',
               name='conv' + str(len(channels) + 1))(x)
    return x


def residual_block(input_tensor, model_type, kernels, filters, block, bn_allowed, stage='a'): 
    # print('Residual block input shape: {}, kernels: {}, filters: {}'.format(K.int_shape(input_tensor),
    #                                                                         kernels,
    #                                                                         filters))
    assert len(filters) == len(kernels), 'Number of filters and number of kernels differs.'
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_name_base = 'bn_' + stage + block + '_branch_'
    conv_name_base = 'conv_' + stage + block + '_branch_'

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
            print('resblock input tensor int_shape: ', K.int_shape(input_tensor))
            print('resblock input tensor shape: ', K.shape(input_tensor))
            print('input_shape: ', input_tensor.shape)
            padding = K.zeros(K.int_shape(input_tensor), name='padding')
            #print(padding._keras_history)
            input_tensor = Concatenate()([input_tensor, padding])
        if model_type == 'decoder':
            input_tensor = Conv2D(K.int_shape(x)[-1], 1)(input_tensor)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    # print('Residual block output shape: ', K.int_shape(x))
    return x
