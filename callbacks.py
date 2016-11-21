from keras.callbacks import LearningRateScheduler, Callback
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
                 latent_dim, batch_size, original_shape,
                 encoder, generator, sampler, 
                 name,
                 frequency,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.original_shape = original_shape
        self.encoder = encoder
        self.generator = generator
        self.sampler = sampler
        self.name = name
        self.frequency = frequency
        super(imageDisplayCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.frequency != 0:
            return

        vis.displayRandom(15, self.latent_dim, self.sampler, self.generator, self.original_shape, "%s-random-%i" % (self.name, epoch+1), batch_size=self.batch_size)
        vis.displaySet(self.x_test[:self.batch_size], 100, self.model, self.original_shape, "%s-test-%i" % (self.name,epoch+1))
        vis.displaySet(self.x_train[:self.batch_size], 100, self.model, self.original_shape, "%s-train-%i" % (self.name,epoch+1))
        vis.displayInterp(self.x_train, self.x_test, self.batch_size, self.latent_dim, self.original_shape, self.encoder, self.generator, 10, "%s-interp-%i" % (self.name,epoch+1))
