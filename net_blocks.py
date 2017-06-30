from keras.layers import Dense, Reshape, Input, Lambda, Convolution2D, Flatten, merge, Deconvolution2D, Activation, BatchNormalization
from keras.regularizers import l1, l2


def dense_block(dims, wd, use_bn, activation):
    layers = []
    for dim in dims:
        layers.append(Dense(dim, W_regularizer=l2(wd)))
        if use_bn:
            layers.append(BatchNormalization(mode=2)) # TODO think about mode
        layers.append(Activation(activation))
    return layers
