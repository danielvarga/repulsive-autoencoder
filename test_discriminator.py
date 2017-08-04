import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU

import model_dcgan
import net_blocks
import callbacks

np.random.seed(10)

trainSize = 50000
disc_size = "small"
wd = 0.0
use_bn = False
batch_size = 200
dense_dims = [100,100]
activation = "leakyrelu"
clipValue = 0.07
verbose=1
nb_epoch = 3
prefix = "pictures/testdisc"


x_train = 10.0 * np.sign(np.random.randint(0,3, size=trainSize) - 1.5) # -10 with 2/3 probability and 10 with 1/3 probability
y_train = np.ones(shape=x_train.shape)
x_generated = np.random.uniform(-10, 10, size=trainSize) # uniform between -10 and 10
y_generated = - np.ones(shape=x_generated.shape)
xs = np.concatenate([x_train, x_generated])
ys = np.concatenate([y_train, y_generated])

#discriminator_channels = model_dcgan.default_channels("discriminator", disc_size, None)
#disc_layers = model_dcgan.discriminator_layers_wgan(discriminator_channels, wd=wd, bn_allowed=use_bn)
disc_layers = net_blocks.dense_block(dense_dims, wd, use_bn, activation)
disc_layers.append(Dense(1, activation="linear"))


disc_input = Input(batch_shape=[batch_size, 1], name="disc_input")
disc_output = disc_input
for layer in disc_layers:
    disc_output = layer(disc_output)
discriminator = Model(disc_input, disc_output)
discriminator.summary()

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    return - K.mean(y_true * y_pred)

clipper = callbacks.ClipperCallback(disc_layers, clipValue)
optimizer = Adam()
discriminator.compile(optimizer=optimizer, loss=D_loss)

discriminator.fit(xs, ys,
                  verbose=verbose,
                  shuffle=True,
                  epochs = nb_epoch,
                  batch_size=batch_size,
                  callbacks=[clipper]
                  )

test = np.linspace(-10, 10, 10 * batch_size)
test_result = discriminator.predict(test, batch_size=batch_size)

name = prefix + "-output-{}.png".format(clipValue)
print "Saving {}".format(name)
plt.figure()
plt.scatter(test, test_result)
plt.savefig(name)
plt.close()
