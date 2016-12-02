import keras
from keras.activations import relu


def myrelu(x):
    return relu(x, alpha=0.05)
keras.activations.myrelu = myrelu

#activation = myrelu
#activation = "linear"
activation = "relu"
