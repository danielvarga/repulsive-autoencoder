import sys # for FlushCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, Callback
from keras import backend as K
from keras.models import Model
import numpy as np
import vis
import load_models

def get_lr_scheduler(nb_epoch, base_lr, lr_decay_schedule):
    assert lr_decay_schedule == sorted(lr_decay_schedule), "lr_decay_schedule has to be monotonically increasing!"

    def get_lr(epoch):
        ratio = float(epoch+1) / nb_epoch
        multiplier = 1.0
        for etap in lr_decay_schedule:
            if ratio > etap:
                multiplier *= 0.1
            else:
                break
#         print "*** LR multiplier: ", multiplier
        return base_lr * multiplier
    return get_lr

class SaveGeneratedCallback(Callback):
    def __init__(self, generator, sampler, prefix, batch_size, frequency, latent_dim, dataset=None, sample_size=100000, save_histogram=False, **kwargs):
        self.generator = generator
        self.sampler = sampler
        self.prefix = prefix
        self.batch_size = batch_size
        self.frequency = frequency
        self.latent_dim = latent_dim
        self.dataset = dataset
        self.sample_size = sample_size
        self.save_histogram = save_histogram
        super(SaveGeneratedCallback, self).__init__(**kwargs)

    def save(self, iteration):
        latent_sample = self.sampler(self.sample_size, self.latent_dim)
        generated = self.generator.predict(latent_sample, batch_size = self.batch_size)
        file = "{}_generated_{}.npy".format(self.prefix, iteration)
        print("Saving generated samples to {}".format(file))
        np.save(file, generated)
        if self.save_histogram and self.dataset is not None:
            vis.print_nearest_histograms(self.dataset, file)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.frequency == 0:
            self.save(epoch+1)

class ImageDisplayCallback(Callback):
    def __init__(self, 
                 x_train, x_test, args,
                 modelDict, sampler,
                 anchor_indices,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.args = args

        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.modelDict = modelDict
        self.ae = modelDict.ae
        self.encoder = modelDict.encoder
        self.encoder_var = modelDict.encoder_var
        self.is_sampling = args.sampling
        self.generator = modelDict.generator
        self.sampler = sampler
        self.anchor_indices = anchor_indices
        self.name = args.callback_prefix
        self.frequency = args.frequency
        self.latent_normal = np.random.normal(size=(self.x_train.shape[0], self.latent_dim))
        self.randomPoints = sampler(args.batch_size, args.latent_dim)
        super(ImageDisplayCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.frequency != 0:
            return

        #print(self.encoder_var.predict(self.x_train, batch_size = self.batch_size))

        randomImages = self.generator.predict(self.randomPoints, batch_size=self.batch_size)
        vis.plotImages(randomImages, 10, self.batch_size // 10, "{}-random-{}".format(self.name, epoch+1))
        vis.plotImages(randomImages, 10, self.batch_size // 10, "{}-random".format(self.name))

        trainImages = self.ae.predict(self.x_train[:self.batch_size], batch_size = self.batch_size)
        images = vis.mergeSets((trainImages, self.x_train[:self.batch_size]))
        vis.plotImages(images, 2 * 10, self.batch_size // 10, "{}-train-{}".format(self.name, epoch+1))
        vis.plotImages(images, 2 * 10, self.batch_size // 10, "{}-train".format(self.name))

        testImages = self.ae.predict(self.x_test[:self.batch_size], batch_size = self.batch_size)
        images = vis.mergeSets((testImages, self.x_test[:self.batch_size]))
        vis.plotImages(images, 2 * 10, self.batch_size // 10, "{}-test-{}".format(self.name, epoch+1))
        vis.plotImages(images, 2 * 10, self.batch_size // 10, "{}-test".format(self.name))

        vis.displayInterp(self.x_train, self.x_test, self.batch_size, self.latent_dim, self.encoder, self.encoder_var, self.is_sampling, self.generator, 10,
            "%s-interp-%i" % (self.name,epoch+1), anchor_indices=self.anchor_indices, toroidal=self.args.toroidal)
        if self.encoder != self.encoder_var:
            vis.plotMVVM(self.x_train, self.encoder, self.encoder_var, self.batch_size, "{}-mvvm-{}.png".format(self.name, epoch+1))
        vis.plotMVhist(self.x_train, self.encoder, self.batch_size, "{}-mvhist-{}.png".format(self.name, epoch+1))
        vis.displayGaussian(self.args, self.modelDict, self.x_train, "{}-dots-{}".format(self.name, epoch+1))
        vis.displayGaussianDots(self.args, self.modelDict, self.x_train, "{}-singledots-{}".format(self.name, epoch+1), 15, 20)

        latent_cloud = self.encoder.predict(self.x_test, batch_size=self.batch_size)

        filename = "{}_latent_mean_{}.npy".format(self.name, epoch+1)
        print("Saving latent pointcloud mean to {}".format(filename))
        np.save(filename, latent_cloud)

        latent_cloud_log_var = self.encoder_var.predict(self.x_test, batch_size=self.batch_size)
        filename = "{}_latent_log_var_{}.npy".format(self.name, epoch+1)
        print("Saving latent pointcloud log variance to {}".format(filename))
        np.save(filename, latent_cloud_log_var)

        if self.latent_dim >= 2:
            if self.args.toroidal:
                assert self.latent_dim == 4
                latent_space_type = "2d_torus_projected"
            elif self.args.spherical:
                if self.latent_dim == 3:
                    latent_space_type = "3d_sphere"
                else:
                    print("visualizing the first two dimensions of a spherical latent space")
                    latent_space_type = "normal"
            else:
                latent_space_type = "normal"
            vis.latentScatter(latent_cloud, "{}-latent-{}".format(self.name, epoch+1), latent_space_type=latent_space_type)
            if self.args.toroidal:
                rot = np.arctan2(latent_cloud[:, 0], latent_cloud[:, 1])
                plt.clf()
                n, bins, patches = plt.hist(rot, 50, normed=1)
                plt.savefig("{}-latenthist-{}.png".format(self.name, epoch+1))

        # TODO move these to vis.py
        if self.latent_dim == 2:
            grid_size = 30
            x = np.linspace(-1.1, +1.1, grid_size)
            y = x
            xx, yy = np.meshgrid(x, y)
            plane = np.vstack([ xx.reshape(-1), yy.reshape(-1) ]).T.reshape((grid_size, grid_size, -1))
            vis.displayPlane(x_train=self.x_train, latent_dim=self.latent_dim, plane=plane,
                         generator=self.generator, name="{}-plane-{}".format(self.name, epoch+1),
                         batch_size=self.batch_size)

        if self.args.toroidal and self.latent_dim == 4:
            grid_size = 30
            x = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
            y = x
            xx, yy = np.meshgrid(x, y)
            # cxc as in circle x circle, a Descartes product
            cxc = np.vstack([ xx.reshape(-1), yy.reshape(-1) ]).T
            assert cxc.shape == (grid_size*grid_size, 2)
            # turning two points on the circle into a point of the torus embedded in 4d:
            embedded = np.vstack([np.cos(cxc[:, 0]), np.sin(cxc[:, 0]), np.cos(cxc[:, 1]), np.sin(cxc[:, 1])]).T
            assert embedded.shape == (grid_size*grid_size, 4)
            plane = embedded.reshape((grid_size, grid_size, 4))
            vis.displayPlane(x_train=self.x_train, latent_dim=self.latent_dim, plane=plane,
                         generator=self.generator, name="{}-flattorus-{}".format(self.name, epoch+1),
                         batch_size=self.batch_size)

        # vis.displayGaussian(self.args, self.ae, self.x_train, "%s-dots-%i" % (self.name, epoch+1))
        # vis.displayRandom(10, self.x_train, self.latent_dim, self.sampler, self.generator, "%s-random-%i" % (self.name, epoch+1), batch_size=self.batch_size)
        # vis.displayRandom(10, self.x_train, self.latent_dim, self.sampler, self.generator, "%s-random" % (self.name), batch_size=self.batch_size)
        # vis.displaySet(self.x_test[:self.batch_size], self.batch_size, self.batch_size, self.model, "%s-test-%i" % (self.name,epoch+1))
        # vis.displaySet(self.x_train[:self.batch_size], self.batch_size, self.batch_size, self.model, "%s-train-%i" % (self.name,epoch+1))

        # count = 5 * self.batch_size
        # generated_samples = self.generator.predict(self.latent_normal[:count], batch_size = self.batch_size)
        # emd = vis.dataset_emd(self.x_train[:count], generated_samples)
        # print "Earth Mover distance between real and generated images: {}".format(emd)

        # latent_train_mean = self.encoder.predict(self.x_train[:count], batch_size = self.batch_size)
        # if self.is_sampling:
        #     latent_train_logvar = self.encoder_var.predict(self.x_train[:count], batch_size = self.batch_size)
        #     latent_train = np.random.normal(size=latent_train_mean.shape) * np.exp(latent_train_logvar/2) + latent_train_mean
        # else:
        #     latent_train = latent_train_mean
        # emd = vis.dataset_emd(self.latent_normal[:count], latent_train)
        # print "Earth Mover distance between latent points and standard normal: {}".format(emd)

class WeightSchedulerCallback(Callback):
    # weight should be a Keras variable
    def __init__(self, nb_epoch, name, startValue, stopValue, start, stop, weight, **kwargs):
        self.nb_epoch = nb_epoch
        self.name = name
        self.weight = weight
        self.startValue = startValue
        self.stopValue = stopValue
        self.start = start
        self.stop = stop
        super(WeightSchedulerCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):        
        phase = 1.0 * (epoch+1) / self.nb_epoch        
        if phase <= self.start:
            relative_phase = 0
        elif phase >= self.stop:
            relative_phase = 1
        else:
            relative_phase = (phase - self.start) / (self.stop - self.start)

        K.set_value(self.weight, (1-relative_phase) * self.startValue + relative_phase * self.stopValue)
        print("\nPhase {}, {} weight: {}".format(phase, self.name, K.eval(self.weight)))

class SaveModelsCallback(Callback):
    def __init__(self, ae, encoder, encoder_var, generator, prefix, frequency, **kwargs):
        self.ae = ae
        self.encoder = encoder
        self.encoder_var = encoder_var
        self.generator = generator
        self.prefix = prefix
        self.frequency = frequency
        super(SaveModelsCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):        
        if (epoch+1) % self.frequency != 0: return        
        load_models.saveModel(self.ae, self.prefix + "_model")
        load_models.saveModel(self.encoder, self.prefix + "_encoder")
        load_models.saveModel(self.encoder_var, self.prefix + "_encoder_var")
        load_models.saveModel(self.generator, self.prefix + "_generator")


class FlushCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        sys.stdout.flush()

class CollectActivationCallback(Callback):
    def __init__(self, nb_epoch, frequency, batch_size, batch_per_epoch, network, trainSet, testSet, layerIndices, prefix, **kwargs):
        self.frequency = frequency
        self.batch_size = batch_size
        self.network = network
        self.trainSet = trainSet
        self.testSet = testSet
        self.prefix = prefix
        self.layerIndices = layerIndices
        self.savedTrain = []
        self.savedTest = []

        self.iterations = batch_per_epoch * nb_epoch / frequency
        self.batch_count = 0

        outputs = []
        for i in range(len(self.network.layers)):
            if i in self.layerIndices:
                output = self.network.layers[i].output
                outputs.append(output)
        self.activation_model = Model([self.network.layers[0].input], outputs)
        train_activations = self.activation_model.predict([self.trainSet], batch_size=self.batch_size)
        test_activations = self.activation_model.predict([self.testSet], batch_size=self.batch_size)

        for train_activation, test_activation in zip(train_activations, test_activations):
            self.savedTrain.append(np.zeros([self.iterations] + list(train_activation.shape)))
            self.savedTest.append(np.zeros([self.iterations] + list(test_activation.shape)))
        super(CollectActivationCallback, self).__init__(**kwargs)

    def on_batch_begin(self, batch, logs):
        if self.batch_count % self.frequency == 0:
            train_activations = self.activation_model.predict([self.trainSet], batch_size=self.batch_size)
            test_activations = self.activation_model.predict([self.testSet], batch_size=self.batch_size)
            for i in range(len(self.layerIndices)):
                self.savedTrain[i][self.batch_count // self.frequency] = train_activations[i]
                self.savedTest[i][self.batch_count // self.frequency] = test_activations[i]
        self.batch_count +=1

    def on_train_end(self, logs):
        fileName = "{}_{}.npz".format(self.prefix, self.frequency)
        outDict = {"train":self.trainSet, "test":self.testSet}
        for i in range(len(self.layerIndices)):
            outDict["train-{}".format(self.layerIndices[i])] = self.savedTrain[i]
            outDict["test-{}".format(self.layerIndices[i])] = self.savedTest[i]
        print("Saving activation history to file {}".format(fileName))
        np.savez(fileName, **outDict)

class ClipperCallback(Callback):
    def __init__(self, layers, clipValue):
        self.layers = layers
        self.clipValue = clipValue

    def on_batch_begin(self, batch, logs):
        self.clip()

    def clip(self):
        if self.clipValue == 0: return
        for layer in self.layers:
#            if layer.__class__.__name__ not in ("Convolution2D"): continue
#            if layer.__class__.__name__ not in ("BatchNormalization"): continue
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = np.clip(weights[i], - self.clipValue, self.clipValue)
            layer.set_weights(weights)

class DiscTimelineCallback(Callback):
    def __init__(self, test_points, batch_size):
        #self.layers = ayers
        #self.discriminator = self.model
        self.test_points = test_points
        self.batch_size = batch_size
        self.timeline = []
        #self.filename = filename

    def on_epoch_end(self, epoch, logs):
        self.timeline.append(self.model.predict(self.test_points, batch_size=self.batch_size))
    
#    def on_epoch_end(self, epoch, logs):
#        self.saveimg(epoch)

