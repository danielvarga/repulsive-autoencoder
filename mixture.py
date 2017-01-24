import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation
from keras.optimizers import SGD, Adam, RMSprop

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

from PIL import Image

def get_param_count(learn_variance, learn_density):
    GAUSS_PARAM_COUNT = 5
    if not learn_variance: GAUSS_PARAM_COUNT -= 2
    if not learn_density: GAUSS_PARAM_COUNT -= 1
    return GAUSS_PARAM_COUNT

class MixtureLayer(Layer):
    def __init__(self, sizeX, sizeY, channel=1, learn_variance=True, learn_density=False, variance=1.0/200, maxpooling=True, **kwargs):
        self.output_dim = 2
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.learn_variance = learn_variance
        self.learn_density = learn_density
        self.variance = variance
        self.maxpooling = maxpooling

        self.xs_index = 0
        self.ys_index = 1
        if learn_variance:
            self.xv_index = 2
            self.yv_index = 3
            if learn_density:
                self.densities_index = 4
        else:
            if learn_density:
                self.densities_index = 2

        super(MixtureLayer, self).__init__(**kwargs)


    # input_shape = (batch, channels, dots, GAUSS_PARAM_COUNT)
    def build(self, input_shape):
        assert len(input_shape) == 4
#        assert input_shape[3] == self.GAUSS_PARAM_COUNT # x, y, xv, yv, density but the last three could be missing!!!
        self.k = input_shape[2] # number of dots to place
        super(MixtureLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inp, mask=None):
        k = self.k
        sizeX = self.sizeX
        sizeY = self.sizeY

        def add_two_dims(t):
            return K.expand_dims(K.expand_dims(t))

        xs = inp[:, :, :, self.xs_index]
        ys = inp[:, :, :, self.ys_index]
        xse = add_two_dims(xs)
        yse = add_two_dims(ys)
        if self.learn_variance:
            xv = inp[:, :, :, self.xv_index]
            yv = inp[:, :, :, self.yv_index]
            xve = add_two_dims(xv)
            yve = add_two_dims(yv)
        if self.learn_density:
            densities = inp[:, :, :, self.xv_index]
            de  = add_two_dims(densities)
        else:
            print "FIXED DENSITY FOR MIXTURE GAUSSIANS!"
            de = 1.0

        xi = tf.linspace(0.0, 1.0, sizeX)
        xi = tf.reshape(xi, [1, 1, 1, -1, 1])
        xi = tf.tile(xi, [1, 1, k, 1, sizeY])
        # -> xi.shape==(1, k, sizeX, sizeY), xi[0][0] has #sizeX different rows, each col has #sizeY identical numbers in it.
        yi = tf.linspace(0.0, 1.0, sizeY)
        yi = tf.reshape(yi, [1, 1, 1, 1, -1])
        yi = tf.tile(yi, [1, 1, k, sizeX, 1])
        

        if self.learn_variance:
            error = (xi - xse) ** 2 / (xve * self.variance) + (yi - yse) ** 2 / (yve * self.variance)
        else:
            error = (xi - xse) ** 2 / self.variance + (yi - yse) ** 2 / self.variance
        error /= 2
        error = tf.minimum(error, 1)
        error = tf.maximum(error, -1)

        # avgpooling is better for reconstruction (if negative ds are allowed),
        # val_loss: 0.0068, but way-way worse for interpolation, it looks like a smoke monster.
        # Note that fixed variance maxpooling will never generalize beyond MNIST.
        if self.maxpooling:
            out = K.max(de * K.exp(-error), axis=2)
        else:
            out = K.sum((2 * de - 1) * K.exp(-error), axis=2)
        out = tf.transpose(out, [0, 2, 3, 1])
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.sizeX, self.sizeY, input_shape[1])



def test_forward():
    inp = np.array([[0.8, 0.8, 0.5], [0.3, 0.2, 0.5]]) # each row specifies a gaussian, as [x, y, density]
    size = 100
    inputs = Input(shape=inp.shape)
    net = MixtureLayer(size,size)(inputs)
    model = Model(input=inputs, output=net)
    out = model.predict([np.expand_dims(inp, 0)])
    out = out[0]
    out = np.clip(out, 0.0, 1.0)
    out *= 255.0

    img = Image.fromarray(out.astype(dtype='uint8'), mode="L")
    img.save("vis.png")


def plotImages(data, n_x, n_y, name):
    height, width = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    mode = "L"
    image_data = np.zeros((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)
    fileName = name + ".png"
    print "Creating file " + fileName
    img.save(fileName)


def displaySet(imageBatch, n, generator, name, flatten_input=False):
    batchSize = imageBatch.shape[0]
    nsqrt = int(np.ceil(np.sqrt(n)))
    if flatten_input:
        net_input = imageBatch.reshape(batchSize, -1)
    else:
        net_input = imageBatch
    recons = generator.predict(net_input, batch_size=batchSize)
    recons = recons.reshape(imageBatch.shape)

    mergedSet = np.zeros(shape=[n*2] + list(imageBatch.shape[1:]))
    for i in range(n):
        mergedSet[2*i] = imageBatch[i]
        mergedSet[2*i+1] = recons[i]
    result = mergedSet.reshape([2*n] + list(imageBatch.shape[1:]))
    plotImages(result, 2*nsqrt, nsqrt, name)


def interpolate(sample_a, sample_b, encoder, decoder, frame_count, output_image_size):
    latent = encoder.predict(np.array([sample_a, sample_b]).reshape(2, -1))
    latent_a, latent_b = latent

    latents = []
    for t in np.linspace(0.0, 1.0, frame_count):
        l = (1-t) * latent_a + t * latent_b
        latents.append(l)
    latents = np.array(latents)
    interp = decoder.predict(latents)
    interp = interp.reshape(frame_count, output_image_size, output_image_size)
    return interp

def test_learn():
    image_size = 28
    nb_features = image_size * image_size
    batch_size = 512
    nb_epoch = 10
    k = 300
    nonlinearity = 'relu'
    intermediate_layer_size = 1000

    # The data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # flatten the 28x28 images to arrays of length 28*28:
    X_train = X_train.reshape(60000, nb_features)
    X_test = X_test.reshape(10000, nb_features)

    # convert brightness values from bytes to floats between 0 and 1:
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    inputs = Input(shape=(nb_features,))
    net = inputs
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(intermediate_layer_size, activation=nonlinearity)(net)
    net = Dense(k * GAUSS_PARAM_COUNT, activation='sigmoid')(net)
    gaussians = Reshape((k, GAUSS_PARAM_COUNT))(net)
    net = MixtureLayer(image_size, image_size)(gaussians)
    # net = Activation('sigmoid')(net)
    net = Reshape((nb_features,))(net)
    model = Model(input=inputs, output=net)

    model.summary()

    model.compile(loss='mse', optimizer=Adam())

    history = model.fit(X_train, X_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, X_test))

    n = 400
    displaySet(X_train[:n].reshape(n, image_size, image_size), n, model, "ae-train", flatten_input=True)
    displaySet(X_test [:n].reshape(n, image_size, image_size), n, model, "ae-test",  flatten_input=True)

    encoder = Model(input=inputs, output=gaussians)
    encoder.compile(loss='mse', optimizer=SGD())

    input_gaussians = Input(shape=(k, GAUSS_PARAM_COUNT))
    output_image_size = image_size * 4 # It's cheap now!
    decoder_layer = MixtureLayer(output_image_size, output_image_size)(input_gaussians)
    decoder_layer = Reshape((output_image_size*output_image_size,))(decoder_layer)
    decoder = Model(input=input_gaussians, output=decoder_layer)
    decoder.compile(loss='mse', optimizer=SGD())

    frame_count = 30
    output_image_size = 4 * image_size

    interp = interpolate(X_train[31], X_train[43], encoder, decoder, frame_count, output_image_size)
    plotImages(interp, 10, 10, "ae-interp")

    anim_phases = 10
    animation = []

    targets = []
    def collect(target_digit, anim_phases):
        i = 0
        j = 0
        while j < anim_phases:
            if y_train[i] == target_digit:
                targets.append(i)
                j += 1
            i += 1
    collect(3, anim_phases)
    collect(5, anim_phases)
    targets += range(anim_phases)
    print "Animation phase count %d" % len(targets)

    for i in range(len(targets)-1):
        interp = interpolate(X_train[targets[i]], X_train[targets[i+1]], encoder, decoder, frame_count, output_image_size)
        animation.extend(interp[:-1])

    print "Creating frames of animation"
    for i, frame_i in enumerate(animation):
        img = Image.fromarray((255 * np.clip(frame_i, 0.0, 1.0)).astype(dtype='uint8'), mode="L")
        img.save("gif/%03d.gif" % i)

# test_learn()
