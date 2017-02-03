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

    def __call__(self, z, y):

	critic_layers = []

        #h = Flatten()(x)
	h = z

	layers = []
        for intermediate_dim in self.intermediate_dims:
            d1 = Dense(intermediate_dim, W_regularizer=l2(self.wd))
	    layers.append(d1)
	    critic_layers.append(d1)
            layers.append(Activation(self.activation))

	d1 = Dense(1, W_regularizer=l2(self.wd))
	layers.append(d1)
	critic_layers.append(d1)


        critic_input = Input(batch_shape=(200,self.latent_dim,))
        critic_output = critic_input
        for layer in layers:
            critic_output = layer(critic_output)
            h = layer(h)

        return critic_input, critic_output, h, critic_layers

