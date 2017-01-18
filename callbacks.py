import sys # for FlushCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, Callback
from keras import backend as K
import numpy as np
import vis

def get_lr_scheduler(nb_epoch, base_lr):

    def get_lr(epoch):
        if epoch < nb_epoch * 0.5:
            return base_lr
        elif epoch < nb_epoch * 0.8:
            return base_lr * 0.1
        else:
            return base_lr * 0.01
    return LearningRateScheduler(get_lr)

class ImageDisplayCallback(Callback):
    def __init__(self, 
                 x_train, x_test, 
                 latent_dim, batch_size,
                 encoder, encoder_var, is_sampling, generator, sampler, 
                 name,
                 frequency,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder = encoder
        self.encoder_var = encoder_var
        self.is_sampling = is_sampling
        self.generator = generator
        self.sampler = sampler
        self.name = name
        self.frequency = frequency
        super(ImageDisplayCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.frequency != 0:
            return

        vis.displayRandom(15, self.x_train, self.latent_dim, self.sampler, self.generator, "%s-random-%i" % (self.name, epoch+1), batch_size=self.batch_size)
        vis.displaySet(self.x_test[:self.batch_size], self.batch_size, self.batch_size, self.model, "%s-test-%i" % (self.name,epoch+1))
        vis.displaySet(self.x_train[:self.batch_size], self.batch_size, self.batch_size, self.model, "%s-train-%i" % (self.name,epoch+1))
        vis.displayInterp(self.x_train, self.x_test, self.batch_size, self.latent_dim, self.encoder, self.encoder_var, self.is_sampling, self.generator, 10, "%s-interp-%i" % (self.name,epoch+1))
        if self.encoder != self.encoder_var:
            vis.plotMVVM(self.x_train, self.encoder, self.encoder_var, self.batch_size, "{}-mvvm-{}.png".format(self.name, epoch+1))
        vis.plotMVhist(self.x_train, self.encoder, self.batch_size, "{}-mvhist-{}.png".format(self.name, epoch+1))


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
        print "\nPhase {}, {} weight: {}".format(phase, self.name, K.eval(self.weight))


class FlushCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        sys.stdout.flush()
