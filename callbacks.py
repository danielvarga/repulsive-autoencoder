import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, Callback
from keras import backend as K
import numpy as np
import vis

def get_lr_scheduler(nb_epoch):

    def get_lr(epoch):
        base_lr = 0.001
        if epoch < nb_epoch * 0.5:
            return base_lr
        elif epoch < nb_epoch * 0.8:
            return base_lr * 0.1
        else:
            return base_lr * 0.01
    return LearningRateScheduler(get_lr)

class imageDisplayCallback(Callback):
    def __init__(self, 
                 x_train, x_test, 
                 latent_dim, batch_size,
                 encoder, encoder_var, generator, sampler, 
                 name,
                 frequency,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder = encoder
        self.encoder_var = encoder_var
        self.generator = generator
        self.sampler = sampler
        self.name = name
        self.frequency = frequency
        super(imageDisplayCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.frequency != 0:
            return

        vis.displayRandom(15, self.x_train, self.latent_dim, self.sampler, self.generator, "%s-random-%i" % (self.name, epoch+1), batch_size=self.batch_size)
        vis.displaySet(self.x_test[:self.batch_size], 100, self.model, "%s-test-%i" % (self.name,epoch+1))
        vis.displaySet(self.x_train[:self.batch_size], 100, self.model, "%s-train-%i" % (self.name,epoch+1))
        vis.displayInterp(self.x_train, self.x_test, self.batch_size, self.latent_dim, self.encoder, self.generator, 10, "%s-interp-%i" % (self.name,epoch+1))
        if self.encoder != self.encoder_var:
            vis.plotMVVM(self.x_train, self.encoder, self.encoder_var, self.batch_size, "{}-mvvm-{}.png".format(self.name, epoch+1))
        vis.plotMVhist(self.x_train, self.encoder, self.batch_size, "{}-mvhist-{}.png".format(self.name, epoch+1))


class weightSchedulerCallback(Callback):
    # weightPrimary and weightSecondary should be Keras variables
    def __init__(self, nb_epoch, weightPrimary, weightSecondary, start, stop, **kwargs):
        self.nb_epoch = nb_epoch
        self.weightPrimary = weightPrimary
        self.weightSecondary = weightSecondary
        self.start = start
        self.stop = stop
        super(weightSchedulerCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        phase = 1.0 * (epoch+1) / self.nb_epoch
        if phase < self.start:
            K.set_value(self.weightPrimary, 1)
            K.set_value(self.weightSecondary, 0)
        elif phase < self.stop:
            K.set_value(self.weightSecondary, (phase-self.start) / (self.stop - self.start))
            K.set_value(self.weightPrimary, (self.stop - phase) / (self.stop - self.start))
        else:
            K.set_value(self.weightPrimary, 0)
            K.set_value(self.weightSecondary, 1)
        print "\n{}: Primary - secondary weights: ({} - {})".format(phase, K.eval(self.weightPrimary), K.eval(self.weightSecondary))
        
