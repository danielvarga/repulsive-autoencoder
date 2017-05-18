import sys
import numpy as np
from keras.layers import Input, Flatten, Dense, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD
import keras.backend as K
import tensorflow as tf
import time
import sklearn
import sklearn.neighbors
import sklearn.linear_model

import params
import data
import vis
import callbacks
import samplers
import kohonen
import transform_images

import model_dcgan
import dense

args = params.getArgs()
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

def display_elapsed(startTime, endTime):
    elapsed = endTime - startTime
    second = elapsed % 60
    minute = int(elapsed / 60)
    print "Elapsed time: {}:{:.0f}".format(minute, second)

print "loading data"
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
x_true_flow = data_object.get_train_flow(args.batch_size)
args.original_shape = x_train.shape[1:]
data_count = len(x_train)

if args.dataset == "celeba":
    label_names, labels = data_object.get_labels()
    labels = labels[:args.trainSize]

def project_to_sphere(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)

latent_unnormed = samplers.gaussian_sampler(data_count, args.latent_dim)

if args.spherical:
    sampler = samplers.spherical_sampler
    latent = project_to_sphere(latent_unnormed)
else:
    sampler = samplers.gaussian_sampler
    latent = latent_unnormed
    
if args.oversampling > 0:
    assert args.ornstein == 1.0, "Do not use oversampling and ornsteing dynamics together."
    extra_latent = sampler(args.oversampling, args.latent_dim)

if args.use_augmentation:
    assert not args.use_labels_as_latent, "I don't know how use_augmentation and use_labels_as_latent should work together"
    latent[:, :transform_images.transformation_types] = np.zeros((data_count, transform_images.transformation_types))

if args.latent_point_file is not None:
    latent = np.load(args.latent_point_file)

if args.use_labels_as_latent:
    assert args.dataset == "celeba" # only celeba has labels
    assert args.ornstein == 1.0 # do not alter latent points
    assert latent.shape == labels.shape
    latent = np.abs(latent) * labels


if args.generator == "dcgan":
    generator_channels = model_dcgan.default_channels("generator", args.dcgan_size, args.original_shape[2])
    reduction = 2 ** (len(generator_channels)+1)
    assert args.original_shape[0] % reduction == 0
    assert args.original_shape[1] % reduction == 0
    gen_firstX = args.original_shape[0] // reduction
    gen_firstY = args.original_shape[1] // reduction
    gen_layers = model_dcgan.generator_layers_wgan(generator_channels, args.latent_dim, args.generator_wd, args.use_bn_gen, args.batch_size, gen_firstX, gen_firstY)
elif args.generator == "dense":
    gen_layers = dense.decoder_layers(args.intermediate_dims, args.original_shape, args.activation, args.generator_wd, args.use_bn_gen)
elif args.generator == "nat6":
#    assert args.dataset == "mnist"
#    assert args.shape == (28,28)
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
        
if args.optimizer == "adam":
    optimizer_g = Adam(lr=args.lr)
elif args.optimizer == "rmsprop":
    optimizer_g = RMSprop(lr=args.lr)
elif args.optimizer == "sgd":
    if args.nesterov > 0:
        optimizer_g = SGD(lr=args.lr, nesterov=True, momentum=args.nesterov)
    else:
        optimizer_g = SGD(lr=args.lr)

if args.modelPath is None:
    gen_input = Input(batch_shape=(args.batch_size,args.latent_dim), name="gen_input")
    # build generator
    gen_output = gen_input
    for layer in gen_layers:
        gen_output = layer(gen_output)
    generator = Model(input=gen_input, output=gen_output)
else:
    print "Loading generator from " +args.modelPath
    generator = vis.loadModel(args.modelPath + "_generator")

generator.compile(optimizer=optimizer_g, loss="mse")
print "Generator:"
generator.summary()

def biflatten(x):
    assert len(x.shape)>=2
    return x.reshape(x.shape[0], -1)

def averageDistance(dataBatch, fakeBatch):
    return np.mean(np.linalg.norm(biflatten(dataBatch) - biflatten(fakeBatch), axis=1))


# callback for saving generated images
generated_saver = callbacks.SaveGeneratedCallback(generator, sampler, args.prefix, args.batch_size, args.frequency, args.latent_dim)

masterPermutation = np.arange(data_count).astype(np.int32)


pairingIndices_list, pairingFakeData_list, pairingRealData_list = [], [], []

def check_separability(data, labels, test_type, n_neighbors=1):
    if test_type == "knn":
        classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='auto')
    elif test_type == "logreg":
        classifier = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    else:
        assert False, "unknown separability test type"

    trainSize = int(len(data) * 0.7)
    classifier.fit(data[:trainSize], labels[:trainSize])
    predict = classifier.predict(data[trainSize:])
    result = float(np.sum(predict == labels[trainSize:])) / len(predict)
    return result

def check_all_separability():
    sample_num = 1000
    for i, name in enumerate(label_names):
        result = check_separability(latent[masterPermutation][:sample_num], labels[:sample_num,i], "knn", 5)
        separability_results[name]["knn5"].append(result)
        result = check_separability(latent[masterPermutation][:sample_num], labels[:sample_num,i], "logreg")
        separability_results[name]["logreg"].append(result)

if args.dataset == "celeba":
    separability_results = {}
    for name in label_names:
        separability_results[name] = {"knn5":[], "logreg":[]} 
    check_all_separability()


randomPoints = sampler(args.batch_size, args.latent_dim)
minibatchCount = data_count // args.batch_size
assert (data_count % args.min_items_in_matching) == 0
assert (args.min_items_in_matching % args.batch_size) == 0
startTime = time.clock()
sys.stderr.write('Starting training\n')
for epoch in range(1, args.nb_iter+1):
    if (epoch % 50 == 0):
        fake = generator.predict(latent[masterPermutation], batch_size = args.batch_size)
        file = "{}_fake_{}.npy".format(args.prefix, epoch)
        print "Saving matched fake pairs to {}".format(file)
        np.save(file, fake)

    allIndices = np.random.permutation(data_count)
    epochDistances = []
    fixedPointRatios = []
    extraLatentRatios = []
    midibatchCount = data_count // args.min_items_in_matching
    for j in range(midibatchCount):
        dataMidiIndices = allIndices[j*args.min_items_in_matching:(j+1)*args.min_items_in_matching]
        assert args.min_items_in_matching==len(dataMidiIndices)

        # find match real and generated images
        dataMidibatch = x_train[dataMidiIndices]
        latentMidibatch = latent[masterPermutation[dataMidiIndices]]
        if args.oversampling > 0:
            latentMidibatch = np.concatenate([latentMidibatch, extra_latent])
        fakeMidibatch = generator.predict(latentMidibatch, batch_size = args.batch_size)

        if args.sampling:
            assert False, "Let's not accidentally hit this codepath."
            sigma = 0.001
            latentMidibatch += np.random.normal(size=latentMidibatch.shape, scale=sigma)
            if args.spherical:
                latentMidibatch = project_to_sphere(latentMidibatch)


        if epoch % args.matching_frequency == 0:
            

            # do the pairing and bookkeep it
            distanceMatrix = kohonen.distanceMatrix(biflatten(fakeMidibatch), biflatten(dataMidibatch), projection=args.projection)
            if args.greedy_matching:
                midibatchPermutation = np.array(kohonen.greedyPairing(biflatten(fakeMidibatch), biflatten(dataMidibatch), distanceMatrix))
            else:
                midibatchPermutation = np.array(kohonen.optimalPairing(biflatten(fakeMidibatch), biflatten(dataMidibatch), distanceMatrix))

            if args.oversampling > 0:
                extra_latent_indices = np.setdiff1d(range(len(latentMidibatch)), midibatchPermutation, assume_unique=True)
                latent[masterPermutation[dataMidiIndices]] = latentMidibatch[midibatchPermutation] # the new latent pairs
                extra_latent = latentMidibatch[extra_latent_indices] # the new extra points
                extraLatentRatio = float(np.sum(midibatchPermutation >= args.min_items_in_matching)) / args.min_items_in_matching
                extraLatentRatios.append(extraLatentRatio)
            else:
                masterPermutation[dataMidiIndices] = masterPermutation[dataMidiIndices[midibatchPermutation]]

            fixedPointRatio = float(np.sum(midibatchPermutation == np.arange(midibatchPermutation.shape[0]))) / midibatchPermutation.shape[0]
            fixedPointRatios.append(fixedPointRatio)
                
            if epoch <= args.no_update_epochs:
                print "midibatch {}, fixedPointRatio {}".format(j, fixedPointRatio)
                sys.stdout.flush()

            latentMidibatch = latent[masterPermutation[dataMidiIndices]] # recalculated

        # collect statistics
        minibatchDistances = averageDistance(dataMidibatch, fakeMidibatch[:args.min_items_in_matching])
        epochDistances.append(minibatchDistances)

        # perform gradient descent to make matched images more similar
        if epoch > args.no_update_epochs:
            gen_loss = generator.fit(latentMidibatch, dataMidibatch, nb_epoch=1, batch_size=args.batch_size, verbose=0)
            if args.use_augmentation:
                offsets = samplers.gaussian_sampler(args.min_items_in_matching, transform_images.transformation_types)
                dataMidibatch = transform_images.shift(dataMidibatch, offsets)
                latentMidibatch[:,:transform_images.transformation_types] = offsets
                gen_loss = generator.fit(latentMidibatch, dataMidibatch, nb_epoch=1, batch_size=args.batch_size, verbose=0)

    epochDistances = np.array(epochDistances)
    epochInterimMean = epochDistances.mean() if len(epochDistances) > 0 else 0.0
    epochInterimMedian = np.median(epochDistances) if len(epochDistances) > 0 else 0.0
    fixedPointRatios = np.array(fixedPointRatios)
    epochFixedPointRatio = np.mean(fixedPointRatios) if len(fixedPointRatios) > 0 else 0.0
    extraLatentRatios = np.array(extraLatentRatios)
    epochExtraLatentRatio = np.mean(extraLatentRatios) if len(extraLatentRatios) > 0 else 0.0

    if args.ornstein < 1.0:
        epsilon = np.random.normal(size=latent.shape)
        latent_unnormed = args.ornstein * latent_unnormed + np.sqrt(1- args.ornstein**2) * epsilon
        if args.spherical:
            latent = project_to_sphere(latent_unnormed)
        else:
            latent = latent_unnormed

    print "epoch %d epochFixedPointRatio %f epochInterimMean %f epochInterimMedian %f epochExtraLatentRatio %f" % (epoch, epochFixedPointRatio, epochInterimMean, epochInterimMedian, epochExtraLatentRatio)
    sys.stdout.flush()
    if epoch % args.frequency == 0:
        currTime = time.clock()
        display_elapsed(startTime, currTime)

        # vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random-{}".format(args.prefix, epoch), batch_size=args.batch_size)
        # vis.displayRandom(10, x_train, args.latent_dim, sampler, generator, "{}-random".format(args.prefix), batch_size=args.batch_size)
        randomImages = generator.predict(randomPoints, batch_size=args.batch_size)
        vis.plotImages(randomImages, 10, args.batch_size // 10, "{}-random-{}".format(args.prefix, epoch))
        vis.plotImages(randomImages, 10, args.batch_size // 10, "{}-random".format(args.prefix))

        trainImages = generator.predict(latent[masterPermutation[:args.batch_size]], batch_size = args.batch_size)
        images = vis.mergeSets((trainImages, x_train[:args.batch_size]))        
        vis.plotImages(images, 2 * 10, args.batch_size // 10, "{}-train-{}".format(args.prefix, epoch))
        vis.plotImages(images, 2 * 10, args.batch_size // 10, "{}-train".format(args.prefix))

        vis.interpBetween(latent[masterPermutation[0]], latent[masterPermutation[1]], generator, args.batch_size, args.prefix + "_interpBetween-{}".format(epoch))
        vis.interpBetween(latent[masterPermutation[0]], latent[masterPermutation[1]], generator, args.batch_size, args.prefix + "_interpBetween")
        vis.saveModel(generator, args.prefix + "_generator")

        file = "{}_latent.npy".format(args.prefix)
        print "Saving latent points to {}".format(file)
        np.save(file, latent[masterPermutation])

        if args.oversampling:
            file = "{}_extra_latent.npy".format(args.prefix)
            print "Saving extra latent points to {}".format(file)
            np.save(file, extra_latent)

    if epoch % 200 == 0:
        vis.saveModel(generator, args.prefix + "_generator_{}".format(epoch))
        generated_saver.save(epoch)

if args.dataset == "celeba":
    check_all_separability()
    print "separability results for greater than 1 % change"
    for name in label_names:
        for k in separability_results[name].keys():
            results = separability_results[name][k]
            change = np.abs(results[0] - results[-1])
            if change > 0.01:
                print "{} {}: {} -> {}".format(name, k, results[0], results[-1])
