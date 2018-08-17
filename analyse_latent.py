from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

import numpy as np
import os

prefix = "/home/daniel/experiments/repulsive-autoencoder/pictures/dcgan_vae_lsun/dcgan_vae_lsun_latent"

meanFile = prefix + "_mean_200.npy"
logvarFile = prefix + "_log_var_200.npy"

mean = np.load(meanFile)
logvar = np.load(logvarFile)
var = np.exp(logvar)
std = np.exp(logvar/2)

print "Mean shape: ", mean.shape
print "Var shape: ", var.shape

# save images into directory ./latent
prefix = "latent"
if not os.path.exists(prefix):
    os.makedirs(prefix)

##############################################################
############# mean ###########################################
##############################################################


# plot the mean of variances (x axis) by the variance of mean (y axis)
def plotMVVM(mean, var, name):
    mean_variances = np.var(mean, axis=0)
    variance_means = np.mean(var, axis=0)
    xlim = (-1, 12)
    ylim = (-1, 3)
    plt.figure(figsize=(12,6))
    plt.scatter(mean_variances, variance_means)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    print("Creating file " + name)
    plt.savefig(name)
    plt.close()
plotMVVM(mean, var, prefix + "/mvvm.png")

def plotMVhist(mean, name):
    mean_variances = np.var(mean, axis=0)
    # histogram = np.histogram(mean_variances, 30)
    histogram = np.histogram(mean_variances, bins=(0, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0)) #100, range=(0,3))
    print "MVhist:"
    print histogram
    mean_variances = histogram[1]
    variance_means = [0] + list(histogram[0])
    xlim = (0,np.max(mean_variances))
    ylim = (0,np.max(variance_means))
    plt.figure(figsize=(12,6))
    plt.scatter(mean_variances, variance_means)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    print("Creating file " + name)
    plt.savefig(name)
    plt.close()
plotMVhist(mean, prefix + "/mvhist.png")
    
# visualize the latent mean
def show_latent_cloud(z, name):
    if z.ndim == 1:
        latent_dim = 1
    else:
        latent_dim = z.shape[-1]
    size = z.shape[0]
    if latent_dim==1:
        plt.hist(z, bins=100, density=True)
        # compare it with the standard normal distribution
        x = np.linspace(-3, 3, 100)
        plt.plot(x,mlab.normpdf(x, 0, 1))
    elif latent_dim==2:
        plt.scatter(z[:, 0], z[:, 1])
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z = z[:2000, :]
        ax.scatter(z[:, 0], z[:, 1], z[:, 2])
        ax.set_title('Latent points')
    print("Creating file " + name)
    plt.savefig(name)
    plt.close()
show_latent_cloud(mean, prefix + "/cloud.png")
show_latent_cloud(mean[:,3], prefix + "/cloud1.png")
show_latent_cloud(mean[:,:2], prefix + "/cloud2.png")

# see how far we are from standard normal in the most variable directions
def check_best_normality(mean, prefix, step=10, count=10):
    mean_variances = np.var(mean, axis=0)
    order = np.argsort(mean_variances)[::-1]
    for i in range(count):
        ind = order[i*step]
        selected_mean = mean[:, ind]
        show_latent_cloud(selected_mean, "{}_{}.png".format(prefix, i))
check_best_normality(mean, prefix + "/normality_check", step=10, count=10)


##############################################################
############# sampled ########################################
##############################################################

sampled = np.random.normal(size=var.shape) * std + mean
sampled_mean = np.mean(sampled, axis=0)
sampled_var = np.var(sampled, axis=0)

# plot the mean of variances (x axis) by the variance of mean (y axis) for sampled points
plotMVVM(sampled_mean, sampled_var, prefix + "/sampled_mvvm.png")

# visualize sampled latent points
show_latent_cloud(sampled, prefix + "/sampled_cloud.png")
show_latent_cloud(sampled[:,3], prefix + "/sampled_cloud1.png")
show_latent_cloud(sampled[:,:2], prefix + "/sampled_cloud2.png")


# see how far we are from standard normal in the most variable directions
check_best_normality(sampled, prefix + "/sampled_normality_check", step=10, count=10)
