import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import numpy as np
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Reshape, Flatten
from keras.models import Model
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
import os

import model_dcgan
import net_blocks
import callbacks

np.random.seed(100)

epsilon = 0.00001
trainSize = 50000
disc_size = "small"
wd = 0.0
use_bn = True
batch_size = 200
dense_dims = [100,100, 100, 100]
activation = "relu"
clipValue = 0.01
gradient_weight = 100.0
verbose=2
prefix = "pictures/testdisc"
epochs = [10, 10, 10]
dim=2
dim2_anim = True
create_timeline = True

if dim == 1:
    # x_train = 10.0 * np.sign(np.random.randint(0,3, size=trainSize) - 1.5) # -10 with 2/3 probability and 10 with 1/3 probability
    x_values = np.array([-10, 0, 0, 10, 10])
    x_train = x_values[np.random.randint(0,5, size=trainSize)]
    x_generated = np.random.uniform(-10, 10, size=trainSize) # uniform between -10 and 10
    test_points = np.linspace(-15, 15, batch_size)
if dim == 2:
    # x_train = 10.0 * np.sign(np.random.randint(0,3, size=trainSize) - 1.5) # -10 with 2/3 probability and 10 with 1/3 probability
    x_values = np.array([[-10,-10], [-10,-10], [0,0], [0,0], [10,10], [10, 10]])
    x_train = x_values[np.random.randint(0,6, size=trainSize)]
    x_generated = np.random.uniform(-10, 10, size=(trainSize, 2))
    x_coords = np.linspace(-15, 15, batch_size)
    test_points = np.transpose([np.tile(x_coords, len(x_coords)), np.repeat(x_coords, len(x_coords))])

# x_generated = 3.0 * np.random.normal(size=trainSize) + x_train

y_train = np.ones(shape=[len(x_train)])
y_generated = - y_train
xs = np.concatenate([x_train, x_generated])
ys = np.concatenate([y_train, y_generated])

#discriminator_channels = model_dcgan.default_channels("discriminator", disc_size, None)
#disc_layers = model_dcgan.discriminator_layers_wgan(discriminator_channels, wd=wd, bn_allowed=use_bn)
disc_layers = net_blocks.dense_block(dense_dims, wd, use_bn, activation)
disc_layers.append(Dense(1, activation="linear"))


disc_input = Input(batch_shape=[batch_size, dim], name="disc_input")
disc_output = disc_input
for layer in disc_layers:
    disc_output = layer(disc_output)

# y_true = 1 (real_image) or -1 (generated_image)
# we push the real examples up, the false examples down
def D_loss(y_true, y_pred):
    loss = - K.mean(y_true * y_pred)
    return loss
def grad_aux(output, input):
    grads = K.gradients(output, [input])
    grads = grads[0]
    tensor_axes = range(1, K.ndim(grads))
    grads = K.sqrt(epsilon + K.sum(K.square(grads), axis=tensor_axes))
    return grads
def grad_flat(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.square(K.maximum(k1, grads) - k1))
    return grad_penalty
def grad_orig(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.square(grads - k1))
    return grad_penalty
def grad_hill(y_true, y_pred):
    grads = grad_aux(y_pred, disc_input)
    k1 = K.constant(1.0)
    grad_penalty = K.mean(K.abs(K.square(grads) - k1))
    return grad_penalty


def hill(y_true, y_pred):
    return D_loss(y_true, y_pred) + gradient_weight * grad_hill(y_true, y_pred)
def orig(y_true, y_pred):
    return D_loss(y_true, y_pred) + gradient_weight * grad_orig(y_true, y_pred)
def flat(y_true, y_pred):
    return D_loss(y_true, y_pred) + gradient_weight * grad_flat(y_true, y_pred)

metrics = [D_loss, grad_flat, grad_orig, grad_hill]

discriminator = Model(disc_input, disc_output)
optimizer = RMSprop()
discriminator.compile(optimizer=optimizer, loss="mse", metrics=metrics)
discriminator.summary()
discriminator.save_weights("a.h5")

losses = [orig, hill, flat]
epochs = [10, 10, 10]
#losses = [orig]
#epochs = [3]
count = len(losses)
predictions = []

if create_timeline:
    cb = callbacks.DiscTimelineCallback(test_points, batch_size)


for i in range(count):
    loss = losses[i]
    nb_epoch = epochs[i]
    optimizer = RMSprop()
    discriminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    discriminator.load_weights("a.h5")
    discriminator.fit(xs, ys,
                      verbose=verbose,
                      shuffle=True,
                      epochs = nb_epoch,
                      batch_size=batch_size,
                      callbacks=[cb]
    )
    predictions.append(discriminator.predict(test_points, batch_size=batch_size))



if not os.path.exists('pictures/disc_anim_tmp'):
    os.mkdir('pictures/disc_anim_tmp')
        
os.system("rm -f pictures/disc_anim_tmp/*.png")

#  Creating timeline

def save3Dplot(X, Y, Z, file, angle=70, title=''):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')
    colmap = cm.ScalarMappable(cmap=cm.bone)
    Z = np.array(Z).squeeze()
    colmap.set_array((Z))
    
    yg = ax.scatter(X, Y, Z, c=cm.bone(((Z-min(Z)) / max((Z-min(Z)))  )))
    cb = fig.colorbar(colmap)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    ax.view_init(30, angle)
    plt.savefig(file)
    plt.close()


if create_timeline:
    timeline = cb.timeline
    print ('Creating timeline...')
    c_ep = 0
    for id,ep in enumerate(epochs):
        for i in range(ep):
            zs = timeline[c_ep][:,0]
            save3Dplot(test_points[:,0], test_points[:,1], zs, 'pictures/disc_anim_tmp/%02d_%02d.png' % (ep, i), title='%d' % (i+1) )
            #print np.array(zs).shape
            #sys.exit()
            c_ep += 1
        os.system("convert -delay 200 -loop 0 pictures/disc_anim_tmp/*.png pictures/disc_anim_timeline_%d.gif" % (id+1))
        print ("Saved pictures/disc_anim_timeline_%d.gif" % (id+1))
        os.system("rm pictures/disc_anim_tmp/*.png")



# display result
name = prefix + "-output.png"
fig = plt.figure()
for i in range(count):
    if dim == 1:
        
        print "Saving {}".format(name)
        ax = fig.add_subplot(count, 1, i+1)
 #       f, axes = plt.subplots(count, 1, sharex=True)
        ax.scatter(test_points, predictions[i])
        plt.savefig(name)

    elif dim == 2:
        # NOTE: Created 3D images and animations dont look good on geforce1.
        if dim2_anim:
            print ("Creating animation %d of %d..." % (i+1, count))

        #ax = fig.add_subplot(count, 1, i+1, projection='3d')
        #        ax.scatter(test_points[:,0], test_points[:,1], predictions[i])

        xs = test_points[:,0]
        ys = test_points[:,1]
        zs = predictions[i]
        #zs = zs - min(zs)
        
        #np.save('xs_%d.npy' % i, xs)
        #np.save('ys_%d.npy' % i, ys)
        #np.save('zs_%d.npy' % i, zs)


        minangle = 0 if dim2_anim else 30
        maxangle = 360 if dim2_anim  else 31
        for angle in range(minangle, maxangle, 10):
            save3Dplot(xs, ys, zs, 'pictures/disc_anim_tmp/%03d_%03d.png' % (i, angle), angle=angle)
            #fig = plt.figure(figsize=(8,6))

            #ax = fig.add_subplot(111,projection='3d')
            
            #colmap = cm.ScalarMappable(cmap=cm.bone)
            #colmap.set_array((zs))
            
            #yg = ax.scatter(xs, ys, zs, c=cm.bone(((zs-min(zs)) / max((zs-min(zs)))  )[:,0]))
            #cb = fig.colorbar(colmap)

            #ax.set_xlabel('X')
            #ax.set_ylabel('Y')
            #ax.set_zlabel('Z')

            #ax.view_init(30, angle)
            #plt.savefig('pictures/disc_anim_tmp/%03d.png' % angle)
            #plt.close()

        if dim2_anim:
            os.system("convert -delay 20 -loop 10 pictures/disc_anim_tmp/*.png pictures/disc_anim_%d.gif" % (i+1))
            print ("Saved pictures/disc_anim_%d.gif" % (i+1))
            os.system("rm pictures/disc_anim_tmp/*.png")
        else:
            print ("Saved pictures/disc_anim_tmp/%03d_030.png" % i)
