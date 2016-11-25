import keras
from keras.activations import relu


activation = "relu"
def myrelu(x):
    return relu(x, alpha=0.05)
keras.activations.myrelu = myrelu
#activation = myrelu
