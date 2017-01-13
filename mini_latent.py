import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.linalg
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
import os.path

import params
args = params.getArgs()
print(args)



prefix = args.prefix
shape = args.shape
batch_size = args.batch_size
do_latent_variances = args.sampling
color = args.color


latent_train_mean_file = prefix + "_latent_train_mean.npy"
latent_train_logvar_file = prefix + "_latent_train_logvar.npy"
latent_train_file = prefix + "_latent_train.npy"


latent_train_mean = np.load(latent_train_mean_file)

if do_latent_variances:
    assert os.path.isfile(latent_train_logvar_file)
    latent_train_logvar = np.load(latent_train_logvar_file)
    if os.path.isfile(latent_train_file):
        print "reading post-sampling latent cloud from cache", latent_train_file
        latent_train = np.load(latent_train_file)
    else:
        print "calculating post-sampling latent cloud, saving to", latent_train_file
        latent_train = np.random.normal(size=latent_train_mean.shape) * np.exp(latent_train_logvar/2) + latent_train_mean
        np.save(latent_train_file, latent_train)
else:
    latent_train = latent_train_mean


truncate = None
if truncate is not None:
    print "truncating dataset from %d to %d points" % (len(latent_train_mean), truncate)
    latent_train_mean = latent_train_mean[:truncate]
    latent_train_logvar = latent_train_logvar[:truncate]
    latent_train = latent_train[:truncate]


if do_latent_variances:
    x = np.mean(latent_train_logvar, axis=1)
    plt.hist(x, bins = 30)
    plt.savefig(prefix + "_logvar_hist.png")
    plt.close()
    x_indices = np.argsort(x)
#    top10 = x_train[x_indices[:100]]
#    bottom10 = x_train[x_indices[-100:]]
#    xs = np.append(top10, bottom10, axis=0)


print latent_train.shape
origo = np.mean(latent_train, axis=0)
origo_mean = np.mean(latent_train_mean, axis=0)
mean_variances = np.var(latent_train_mean, axis=0)
variances = np.var(latent_train, axis=0)

if do_latent_variances:
    variance_means = np.mean(np.exp(latent_train_logvar), axis=0)
    plt.scatter(mean_variances, variance_means)
    plt.savefig(prefix+"_mvvm.png")
    plt.close()
    print "Variances of means"
    print np.histogram(mean_variances)
    print "Means of variances"
    print np.histogram(variance_means)

    employed_dims_per_sample = np.sum(np.exp(latent_train_logvar) > 0.5, axis=1)
    print "employed_dims_per_sample histogram"
    print np.histogram(employed_dims_per_sample)

    focuses = [] # [4, 11, 5, 12]
    print mean_variances[focuses]
    print variance_means[focuses]
    for focus in focuses:
        print "-------"
        print "focus dim", focus
        print "mean"
        print np.histogram(latent_train_mean[:, focus])
        print "variance"
        print np.histogram(np.exp(latent_train_logvar[:, focus]))


cov_train = np.cov(latent_train_mean.T)
eigVals, eigVects = np.linalg.eig(cov_train)
print "cov_train eigvals = ", sorted(eigVals, reverse=True)

cov_train_sampled = np.cov(latent_train.T)
eigVals_sampled, eigVects_sampled = np.linalg.eig(cov_train_sampled)
print "cov_train_sampled eigvals = ", sorted(eigVals_sampled, reverse=True)



plt.figure()
f, axarr = plt.subplots(4, 4)
focuses = range(16) # [4, 11, 5, 12]

for i in range(16):
    x = i / 4
    y = i % 4
    focus = focuses[i]
    data  = latent_train_mean[focus]
    axarr[x, y].hist(data, bins=100)
    axarr[x, y].set_title("dim %d" % focus)
    axarr[x, y].locator_params(nbins=5, axis='x')
plt.savefig(prefix + "_some_dims.png")
plt.close()


print "premature exit"
sys.exit()


# histogram of the origo
plt.hist(origo, bins = 30)
plt.hist(origo_mean, bins = 30)
plt.savefig(prefix + "_origo.png")
plt.close()

# show mean variances against the location of the origo
plt.scatter(np.absolute(origo), mean_variances)
plt.savefig(prefix + "_origo_variance.png")
plt.close()


# histogram of distances from the origo and from zero
variance = np.mean(np.square(latent_train_mean - origo), axis=1)
variance2 = np.mean(np.square(latent_train_mean), axis=1)
plt.hist(variance, bins = 30, label="Squared istance from mean")
plt.hist(variance2, bins = 30, label="Squared distance from origo")
target = np.random.normal(0.0, 1.0, latent_train_mean.shape)
variance_target = np.mean(np.square(target), axis=1)
plt.hist(variance_target, bins = 30, label="Target squared distance")
plt.legend()
plt.savefig(prefix+"_variance_hist.png")
plt.close()

# histogram of distances from the origo and from zero
sumSquares = np.mean(np.square(latent_train_mean), axis=0)
plt.hist(sumSquares, bins = 30)
plt.savefig(prefix+"_size_contribution.png")
plt.close()
print np.sum(sumSquares)
x1 = np.argmax(sumSquares)
x2 = np.argmin(sumSquares)
plt.figure()
f, axarr = plt.subplots(2, 2)
greatest = latent_train_mean[:, x1]
smallest = latent_train_mean[:, x2]
data = (greatest, smallest, np.square(greatest), np.square(smallest))
titles = ('Greatest dim', 'Smallest dim', 'Greatest dim squared', 'Smallest dim squared')
for i in range(4):
    x = i / 2
    y = i % 2
    axarr[x, y].hist(data[i], bins=100)
    axarr[x, y].set_title(titles[i])
    axarr[x, y].locator_params(nbins=5, axis='x')
plt.savefig(prefix + "_square_contribution.png")
plt.close()



variances = np.var(latent_train, axis=0)
working_mask = (variances > 0.2)
print "Variances"
print np.sum(working_mask), "/", working_mask.shape
print np.histogram(variances, 100)

latent_dim = latent_train.shape[1]


# the below loop illustrates that taking small subsamples will not alter the eigenvalue structure of the covariance matrix 
# for cnt in range(10, 500, 10):
#     latent_sample = latent_train_mean[:cnt]
#     cov_sample = np.cov(latent_sample.T)
#     eigvals_sample = list(np.linalg.eigvals(cov_sample).real)
#     print "cov eigvals using first {} samples:\n".format(cnt), sorted(eigvals_sample, reverse=True)


print "CS", cov_train.shape
std_train = np.std(latent_train_mean)
print "MS", origo_mean.shape
cho = np.linalg.cholesky(cov_train)
print "CHOS", cho.shape
N = 100000
z = np.random.normal(0.0, 1.0, (N, latent_dim))
sample = cho.dot(z.T).T+origo_mean
print sample.shape


def oval_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    z = cho.dot(z.T).T+origo_mean
#    z /= np.linalg.norm(z)
    return z

def diagonal_oval_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    z = std_train * z + origo_mean
    return z

def diagonal_oval_sampler_nomean(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    z = std_train * z
    return z

def eigval1d_grid(grid_size, latent_dim):
    x = np.linspace(-2.0, 2.0, num=grid_size)
    xs = []
    for i in range(grid_size):
        xi = x[i] * eigVects[0] * np.sqrt(eigVals[0]) + origo_mean
        xs.append(xi)
    return np.array(xs)


# elliptic==True samples from the Cholesky projected to the eigenvectors' plane.
# elliptic==False samples from the same thing stretched to a circle.
def eigval2d_grid(grid_size, latent_dim, eigVect1, eigVal1, eigVect2, eigVal2, radius=2.0, elliptic=True):
    x = np.linspace(-radius, radius, num=grid_size)
    xs = []
    for i in range(grid_size):
        for j in range(grid_size):
            d1 = eigVect1 * np.sqrt(eigVal1) * x[i]
            if elliptic:
                d2 = eigVect2 * np.sqrt(eigVal2) * x[j]
            else:
                d2 = eigVect2 * np.sqrt(eigVal1) * x[j] # eigVal1!                
            xi = origo_mean + d1 + d2
            xs.append(xi)
    return np.array(xs).reshape((grid_size, grid_size, latent_dim))

grid_size=25

eigpairs =  [(0, 1), (0, 2), (99, 100), (0, 101)]
eigpairs += [(2, 3), (0, 4), (110, 111), (0, 102)]
for i in reversed(range(len(eigpairs))):
    dim1, dim2 = eigpairs[i]
    if (dim1 >= latent_dim) or (dim2 >= latent_dim):
        del eigpairs[i]
