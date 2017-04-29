import sys
import numpy as np
from keras.layers import Input, Flatten, Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
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


if args.generator == "dcgan":
    generator_channels = model_dcgan.default_channels("generator", args.gen_size, args.original_shape[2])
    reduction = 2 ** (len(generator_channels)+1)
    assert args.original_shape[0] % reduction == 0
    assert args.original_shape[1] % reduction == 0
    gen_firstX = args.original_shape[0] // reduction
    gen_firstY = args.original_shape[1] // reduction
    gen_layers = model_dcgan.generator_layers_wgan(generator_channels, args.latent_dim, args.generator_wd, args.use_bn_gen, args.batch_size, gen_firstX, gen_firstY)
elif args.generator == "dense":
    gen_layers = model_dcgan.generator_layers_dense(args.latent_dim, args.batch_size, args.generator_wd, args.use_bn_gen, args.original_shape)
elif args.generator == "nat6":
    assert args.dataset == "mnist"
    assert args.shape == (28,28)
    gen_layers = []
    gen_layers.append(Dense(600))
    gen_layers.append(LeakyReLU(alpha=0.1))
    gen_layers.append(Dense(600))
    gen_layers.append(LeakyReLU(alpha=0.1))
    gen_layers.append(Dense(np.prod(args.original_shape)))
    gen_layers.append(LeakyReLU(alpha=0.1))

    gen_layers.append(Reshape(args.original_shape))

else:
    assert False, "Invalid generator type"
        
gen_input = Input(batch_shape=(args.batch_size,args.latent_dim), name="gen_input")

if args.optimizer == "adam":
    optimizer_g = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_g = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    if args.nesterov > 0:
        optimizer_g = SGD(lr=args.lr, nesterov=True, momentum=args.nesterov)
    else:
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
minibatchCount = data_count // args.batch_size
startTime = time.clock()

# Unit sphere!!!
#latent = sampler(data_count, args.latent_dim)
latent_unnormed = np.random.normal(size=(data_count, args.latent_dim))
latent = latent_unnormed / np.linalg.norm(latent_unnormed, axis=1, keepdims=True)

masterPermutation = np.arange(data_count).astype(np.int32)


pairingIndices_list, pairingFakeData_list, pairingRealData_list = [], [], []

for epoch in range(1, args.nb_iter+1):
    allIndices = np.random.permutation(data_count)
    epochDistances = []
    fixedPointRatios = []
    for i in range(minibatchCount):
        dataIndices = allIndices[i*args.batch_size:(i+1)*args.batch_size]
        assert args.batch_size==len(dataIndices)

        # find match real and generated images
        dataBatch = x_train[dataIndices]
        latentBatch = latent[masterPermutation[dataIndices]]

        if args.sampling:
            sigma = 0.1
            latentBatch += np.random.normal(size=latentBatch.shape, scale=sigma)
            latentBatch /= np.linalg.norm(latentBatch, axis=1, keepdims=True)

        fakeBatch = generator.predict(latentBatch, batch_size = args.batch_size)


        if epoch % args.matching_frequency == 0:
            # perform optimal matching on pairing data to update masterPermutation

            pairingIndices_list.append(dataIndices)
            pairingFakeData_list.append(fakeBatch)
            pairingRealData_list.append(dataBatch)

            if sum([l.shape[0] for l in pairingIndices_list]) >= args.min_items_in_matching:
                # prepare concatenated and flattened data for the pairing
                pairingIndices = np.concatenate(pairingIndices_list)
                pairingFakeData = np.concatenate(pairingFakeData_list)
                pairingRealData = np.concatenate(pairingRealData_list)
                pairingFakeData_flattened = pairingFakeData.reshape((pairingFakeData.shape[0], -1))
                pairingRealData_flattened = pairingRealData.reshape((pairingRealData.shape[0], -1))

                # do the pairint and bookkeep it
                batchPermutation = np.array(kohonen.optimalPairing(pairingFakeData_flattened, pairingRealData_flattened))
                masterPermutation[pairingIndices] = masterPermutation[pairingIndices[batchPermutation]]

                # empty pairing data
                pairingIndices_list, pairingFakeData_list, pairingRealData_list = [], [], []

                latentBatch = latent[masterPermutation[dataIndices]] # recalculated
        else:
            batchPermutation = np.arange(args.batch_size)

        # perform gradient descent to make matched images more similar
        gen_loss = generator.train_on_batch(latentBatch, dataBatch)

        # collect statistics
        minibatchDistances = averageDistance(dataBatch, fakeBatch)
        fixedPointRatio = float(np.sum(batchPermutation == np.arange(batchPermutation.shape[0]))) / batchPermutation.shape[0]
        epochDistances.append(minibatchDistances)
        fixedPointRatios.append(fixedPointRatio)

    epochDistances = np.array(epochDistances)
    epochInterimMean = epochDistances.mean()
    epochInterimMedian = np.median(epochDistances)
    epochFixedPointRatio = np.mean(np.array(fixedPointRatios))

    if args.ornstein < 1.0:
        epsilon = np.random.normal(size=latent.shape)
        latent_unnormed = args.ornstein * latent_unnormed + np.sqrt(1- args.ornstein**2) * epsilon
        latent = latent_unnormed / np.linalg.norm(latent_unnormed, axis=1, keepdims=True)

    print "epoch %d epochFixedPointRatio %f epochInterimMean %f epochInterimMedian %f" % (epoch, epochFixedPointRatio, epochInterimMean, epochInterimMedian)
    sys.stdout.flush()
    if epoch % args.frequency == 0:
        currTime = time.clock()
        display_elapsed(startTime, currTime)
        vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random-{}".format(args.prefix, epoch), batch_size=args.batch_size)
        vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        images = vis.mergeSets((fakeBatch, dataBatch))
        vis.plotImages(images, 2 * 10, args.batch_size // 10, "{}-train-{}".format(args.prefix, epoch))
        vis.plotImages(images, 2 * 10, args.batch_size // 10, "{}-train".format(args.prefix))
        vis.interpBetween(latent[0], latent[1], generator, args.batch_size, args.prefix + "_interpBetween-{}".format(epoch))
        vis.interpBetween(latent[0], latent[1], generator, args.batch_size, args.prefix + "_interpBetween")
        vis.saveModel(generator, args.prefix + "_generator")
    if epoch % 200 == 0:
        vis.saveModel(generator, args.prefix + "_generator_{}".format(epoch))
        generated_saver.save(epoch)
