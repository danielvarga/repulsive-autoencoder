import numpy as np
from keras.layers import Dense, Flatten, Activation, Reshape, Input
from keras.regularizers import l2

class Critic(object):
    pass

class DenseCritic(Critic):
    def __init__(self, latent_dim, intermediate_dims, activation, wd):
	self.latent_dim = latent_dim
        self.intermediate_dims = intermediate_dims
        self.activation = activation
        self.wd = wd

    def __call__(self, x):
        #h = Flatten()(x)
	h = x

	layers = []
        for intermediate_dim in self.intermediate_dims:
            layers.append(Dense(intermediate_dim, W_regularizer=l2(self.wd)))
            layers.append(Activation(self.activation))

	layers.append(Dense(1, W_regularizer=l2(self.wd)))

	# add climp

        critic_input = Input(shape=(self.latent_dim,))
        critic_output = critic_input
        for layer in layers:
            critic_output = layer(critic_output)
            h = layer(h)

        return critic_input, critic_output, h

