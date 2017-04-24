import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import keras.backend as K
import tensorflow as tf
import time

import wgan_params
import data
import vis
import model_dcgan
import callbacks
import samplers
import kohonen

args = wgan_params.getArgs()
print(args)

# set random seed
np.random.seed(10)


# limit memory usage
import keras
print "Keras version: ", keras.__version__
if keras.backend._BACKEND == "tensorflow":
    import tensorflow as tf
    print "Tensorflow version: ", tf.__version__
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))

print "loading data"
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
x_true_flow = data_object.get_train_flow(args.batch_size)
args.original_shape = x_train.shape[1:]

generator_channels = model_dcgan.default_channels("generator", args.gen_size, args.original_shape[2])

reduction = 2 ** (len(generator_channels)+1)
assert args.original_shape[0] % reduction == 0
assert args.original_shape[1] % reduction == 0
gen_firstX = args.original_shape[0] // reduction
gen_firstY = args.original_shape[1] // reduction

if args.generator == "dcgan":
    gen_layers = model_dcgan.generator_layers_wgan(generator_channels, args.latent_dim, args.generator_wd, args.use_bn_gen, args.batch_size, gen_firstX, gen_firstY)
elif args.generator == "dense":
    gen_layers = model_dcgan.generator_layers_dense(args.latent_dim, args.batch_size, args.generator_wd, args.use_bn_gen, args.original_shape)
else:
    assert False, "Invalid generator type"

gen_input = Input(batch_shape=(args.batch_size,args.latent_dim), name="gen_input")

if args.optimizer == "adam":
    optimizer_g = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_g = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    optimizer_g = SGD(lr=args.lr)

sampler = samplers.spherical_sampler

def display_elapsed(startTime, endTime):
    elapsed = endTime - startTime
    second = elapsed % 60
    minute = int(elapsed / 60)
    print "Elapsed time: {}:{:.0f}".format(minute, second)

# build generator
gen_output = gen_input
for layer in gen_layers:
    gen_output = layer(gen_output)

generator = Model(input=gen_input, output=gen_output)
generator.compile(optimizer=optimizer_g, loss="mse")
print "Generator:"
generator.summary()

def averageDistance(dataBatch, fakeBatch):
    return np.mean(np.linalg.norm(dataBatch - fakeBatch, axis=1))

# callback for saving generated images
generated_saver = callbacks.SaveGeneratedCallback(generator, sampler, args.prefix, args.batch_size, args.frequency, args.latent_dim)

data_count = len(x_train)
minibatchSize = data_count // args.batch_size
startTime = time.clock()

# Unit sphere!!!
latent = sampler(data_count, args.latent_dim)
masterPermutation = np.arange(data_count).astype(np.int32)

for epoch in range(1, args.nb_iter+1):
    allIndices = np.random.permutation(data_count)
    epochDistances = []
    fixedPointRatios = []
    for i in range(minibatchSize):
        dataIndices = allIndices[i*args.batch_size:(i+1)*args.batch_size]
        assert args.batch_size==len(dataIndices)

        # find match real and generated images
        dataBatch = x_train[dataIndices]
        latentBatch = latent[masterPermutation[dataIndices]]
        fakeBatch = generator.predict(latentBatch, batch_size = args.batch_size)

        # perform optimal matching on minibatch to update masterPermutation
        fakeBatch_flattened = fakeBatch.reshape((args.batch_size, -1))
        dataBatch_flattened = dataBatch.reshape((args.batch_size, -1))
        batchPermutation = np.array(kohonen.optimalPairing(fakeBatch_flattened, dataBatch_flattened))
        masterPermutation[dataIndices] = masterPermutation[dataIndices[batchPermutation]]

        # perform gradient descent to make matched images more similar
        latentBatch = latent[masterPermutation[dataIndices]] # recalculated
        gen_loss = generator.train_on_batch(latentBatch, dataBatch)

        # collect statistics
        minibatchDistances = averageDistance(dataBatch, fakeBatch)
        fixedPointRatio = float(np.sum(batchPermutation == np.arange(args.batch_size))) / args.batch_size
        epochDistances.append(minibatchDistances)
        fixedPointRatios.append(fixedPointRatio)
    epochDistances = np.array(epochDistances)
    epochInterimMean = epochDistances.mean()
    epochInterimMedian = np.median(epochDistances)
    epochFixedPointRatio = np.mean(np.array(fixedPointRatios))

    
    print "epoch %d epochFixedPointRatio %f epochInterimMean %f epochInterimMedian %f" % (epoch, epochFixedPointRatio, epochInterimMean, epochInterimMedian)
    if epoch % args.frequency == 0:
        currTime = time.clock()
        display_elapsed(startTime, currTime)
        vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random-{}".format(args.prefix, format), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        latent_samples = sampler(2, args.latent_dim)
        vis.interpBetween(latent_samples[0], latent_samples[1], generator, args.batch_size, args.prefix + "_interpBetween-{}".format(epoch))
        vis.interpBetween(latent_samples[0], latent_samples[1], generator, args.batch_size, args.prefix + "_interpBetween")
        vis.saveModel(generator, args.prefix + "_generator")
    if epoch % 20 == 0:
        vis.saveModel(generator, args.prefix + "_generator_{}".format(epoch))
        generated_saver.save(epoch)
