import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, Callback
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
                 encoder, generator, sampler, 
                 name,
                 frequency,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder = encoder
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

class meanVarPlotCallback(Callback):
    def __init__(self, x_train, batch_size, encoder, encoder_var, name, **kwargs):
        self.x_train = x_train
        self.batch_size = batch_size
        self.encoder = encoder
        self.encoder_var = encoder_var
        self.name = name
        super(meanVarPlotCallback, self).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs):
        latent_train_mean = self.encoder.predict(self.x_train, batch_size = self.batch_size)
        latent_train_logvar = self.encoder_var.predict(self.x_train, batch_size = self.batch_size)
        mean_variances = np.var(latent_train_mean, axis=0)
        variance_means = np.mean(np.exp(latent_train_logvar), axis=0)
        plt.scatter(mean_variances, variance_means)
        plt.xlim(-1, 12)
        plt.ylim(-1, 3)
        plt.savefig("{}_mvvm_{}.png".format(self.name,epoch+1))
        
