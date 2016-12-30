import matplotlib
matplotlib.use('Agg')
import numpy as np
import numpy.linalg
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
import os.path

import vis
import data
import model
import params
args = params.getArgs()
print(args)


# limit memory usage
import keras
if keras.backend._BACKEND == "tensorflow":
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
    set_session(tf.Session(config=config))


prefix = args.prefix
shape = args.shape
batch_size = args.batch_size
do_latent_variances = args.sampling
color = args.color

import data
(x_train, x_test) = data.load("celeba", shape=shape, color=color)

generator = vis.loadModel(prefix + "_generator")

latent_train_mean_file = prefix + "_latent_train_mean.npy"
latent_train_logvar_file = prefix + "_latent_train_logvar.npy"
latent_train_file = prefix + "_latent_train.npy"

useCache = False
if useCache and os.path.isfile(latent_train_file):
    latent_train_mean = np.load(latent_train_mean_file)
else:
    encoder = vis.loadModel(prefix + "_encoder")
    latent_train_mean = encoder.predict(x_train, batch_size = batch_size)
    np.save(latent_train_mean_file, latent_train_mean)
if do_latent_variances:
    if useCache and os.path.isfile(latent_train_logvar_file):
        latent_train_logvar = np.load(latent_train_logvar_file)
    else:
        encoder_var = vis.loadModel(prefix + "_encoder_var")
        latent_train_logvar = encoder_var.predict(x_train, batch_size = batch_size)
        np.save(latent_train_logvar_file, latent_train_logvar)
    if useCache and os.path.isfile(latent_train_file):
        latent_train = np.load(latent_train_file)
    else:
        latent_train = np.random.normal(size=latent_train_mean.shape) * np.exp(latent_train_logvar/2) + latent_train_mean
        np.save(latent_train_file, latent_train)
else:
    latent_train = latent_train_mean

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
    print "Mean variances"
    print np.histogram(mean_variances)    

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
plt.hist(variance, bins = 30)
plt.hist(variance2, bins = 30)
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

def masked_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    return z * working_mask




if do_latent_variances:
    for focus_index in range(5): # Index of a specific sample
        focus_latent_mean = latent_train_mean[focus_index]
        focus_latent_logvar = latent_train_logvar[focus_index]

        def single_gaussian_sampler(batch_size, latent_dim):
            shape = [batch_size] + list(focus_latent_mean.shape)
            return np.random.normal(size=shape) * np.exp(focus_latent_logvar/2) + focus_latent_mean

            vis.displayRandom(n=10, x_train=x_train, latent_dim=latent_dim, sampler=single_gaussian_sampler,
                              generator=generator, name=prefix + "_singlesample%d" % focus_index, batch_size=batch_size)



cov_train = np.cov(latent_train_mean.T)
eigvals = list(np.linalg.eigvals(cov_train).real)
print "cov_train eigvals = ", sorted(eigvals, reverse=True)


# the below loop illustrates that taking small subsamples will not alter the eigenvalue structure of the covariance matrix 
# for cnt in range(10, 500, 10):
#     latent_sample = latent_train_mean[:cnt]
#     cov_sample = np.cov(latent_sample.T)
#     eigvals_sample = list(np.linalg.eigvals(cov_sample).real)
#     print "cov eigvals using first {} samples:\n".format(cnt), sorted(eigvals_sample, reverse=True)


print "CS", cov_train.shape
mean_train = np.mean(latent_train_mean, axis=0)
std_train = np.std(latent_train_mean)
print "MS", mean_train.shape
cho = np.linalg.cholesky(cov_train)
print "CHOS", cho.shape
N = 100000
z = np.random.normal(0.0, 1.0, (N, latent_dim))
sample = cho.dot(z.T).T+mean_train
print sample.shape


def oval_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    z = cho.dot(z.T).T+mean_train
#    z /= np.linalg.norm(z)
    return z

def diagonal_oval_sampler(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    z = std_train * z + mean_train
    return z

def diagonal_oval_sampler_nomean(batch_size, latent_dim):
    z = np.random.normal(size=(batch_size, latent_dim))
    z = std_train * z
    return z

eigVals, eigVects = np.linalg.eig(cov_train)

def eigval1d_grid(grid_size, latent_dim):
    x = np.linspace(-2.0, 2.0, num=grid_size)
    xs = []
    for i in range(grid_size):
        xi = x[i] * eigVects[0] * np.sqrt(eigVals[0]) + mean_train
        xs.append(xi)
    return np.array(xs)


# elliptic==True samples from the Cholesky projected to the eigenvectors' plane.
# elliptic==False samples from the same thing stretched to a circle.
def eigval2d_grid(grid_size, latent_dim, eigIndex1=0, eigIndex2=1, radius=2.0, elliptic=True):
    x = np.linspace(-radius, radius, num=grid_size)
    xs = []
    for i in range(grid_size):
        for j in range(grid_size):
            d1 = eigVects[eigIndex1] * np.sqrt(eigVals[eigIndex1]) * x[i]
            if elliptic:
                d2 = eigVects[eigIndex2] * np.sqrt(eigVals[eigIndex2]) * x[j]
            else:
                d2 = eigVects[eigIndex2] * np.sqrt(eigVals[eigIndex1]) * x[j] # eigIndex1!                
            xi = mean_train + d1 + d2
            xs.append(xi)
    return np.array(xs).reshape((grid_size, grid_size, latent_dim))

grid_size=25

eigpairs =  [(0, 1), (0, 2), (99, 100), (0, 101)]
eigpairs += [(2, 3), (0, 4), (110, 111), (0, 102)]
for i in reversed(range(len(eigpairs))):
    dim1, dim2 = eigpairs[i]
    if (dim1 >= latent_dim) or (dim2 >= latent_dim):
        del eigpairs[i]
    
for eigIndex1, eigIndex2 in eigpairs:
    print "eigenplane grid", eigIndex1, eigIndex2
    plane = eigval2d_grid(grid_size, latent_dim, eigIndex1=eigIndex1, eigIndex2=eigIndex2, radius=4.0, elliptic=True)
    vis.displayPlane(x_train=x_train, latent_dim=latent_dim, plane=plane,
        generator=generator, name=prefix + "_eigs%d-%d" % (eigIndex1, eigIndex2), batch_size=batch_size)
    plane = eigval2d_grid(grid_size, latent_dim, eigIndex1=eigIndex1, eigIndex2=eigIndex2, radius=4.0, elliptic=False)
    vis.displayPlane(x_train=x_train, latent_dim=latent_dim, plane=plane,
        generator=generator, name=prefix + "_eigs-nonell%d-%d" % (eigIndex1, eigIndex2), batch_size=batch_size)

np.random.seed(10)
vis.displayRandom(n=20, x_train=x_train, latent_dim=latent_dim, sampler=masked_sampler,
        generator=generator, name=prefix + "_masked", batch_size=batch_size)

np.random.seed(10)
vis.displayRandom(n=20, x_train=x_train, latent_dim=latent_dim, sampler=model.gaussian_sampler,
                  generator=generator, name=prefix + "_standard", batch_size=batch_size)

np.random.seed(10)
vis.displayRandom(n=20, x_train=x_train, latent_dim=latent_dim, sampler=oval_sampler,
        generator=generator, name=prefix + "_oval", batch_size=batch_size)

np.random.seed(10)
vis.displayRandom(n=20, x_train=x_train, latent_dim=latent_dim, sampler=diagonal_oval_sampler,
        generator=generator, name=prefix + "_diagonal_oval", batch_size=batch_size)

np.random.seed(10)
vis.displayRandom(n=20, x_train=x_train, latent_dim=latent_dim, sampler=diagonal_oval_sampler_nomean,
        generator=generator, name=prefix + "_diagonal_oval_nomean", batch_size=batch_size)




do_tsne = False

if do_tsne:
    from sklearn.manifold import TSNE
    import sklearn
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, metric="euclidean")
    n = 5000
    latent_train_sampled = latent_train[np.random.choice(latent_train.shape[0], size=n, replace=False)]
    # print latent_train_sampled[0, :]
    # print latent_train_sampled[:, 0]
    reduced = tsne.fit_transform(latent_train_sampled)

    plt.figure(figsize=(12,12))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.savefig(prefix + "_tsne.png")
    plt.close()

# to be deleted eventually...

# for i in range(10):
#     indices = np.random.choice(200, 10)
#     vis.plot2Dprojections(latent_train[:], indices, prefix + "_projections_{}.png".format(i))


# projector = GaussianRandomProjection(n_components=2, random_state=81)
# projected_train = projector.fit_transform(latent_train)
# projected_test = projector.fit_transform(latent_test)

#projected_train = latent_train[:, [0,1]]
#projected_test = latent_test[:, [0,1]]

# mymin = np.min((np.min(projected_train), np.min(projected_test)))
# mymax = np.max((np.max(projected_train), np.max(projected_test)))
# dim = np.max(np.abs((mymin,mymax)))

# plt.figure(figsize=(14,6))
# ax1 = plt.subplot(121)
# ax1.hexbin( projected_train[:, 0], projected_train[:, 1])
# plt.xlim(-dim,dim)
# plt.ylim(-dim,dim)
# ax2 = plt.subplot(122)
# ax2.hexbin( projected_test[:, 0], projected_test[:, 1])
# plt.xlim(-dim,dim)
# plt.ylim(-dim,dim)
# plt.savefig(prefix + "_hexbin.png")

# corr_train = np.corrcoef(latent_train.T)
# corr_test = np.corrcoef(latent_test.T)

# plt.figure(figsize=(12,24))
# ax1 = plt.subplot(211)
# ax1.matshow(np.abs(corr_train), cmap='coolwarm')
# ax2 = plt.subplot(212)
# ax2.matshow(np.abs(corr_test), cmap='coolwarm')
# plt.savefig(prefix + "_corr.png")

# corr_learned = np.corrcoef(sample.T)
# plt.figure(figsize=(12,24))
# ax1 = plt.subplot(211)
# ax1.matshow(np.abs(corr_train), cmap='coolwarm')
# ax2 = plt.subplot(212)
# ax2.matshow(np.abs(corr_learned), cmap='coolwarm')
# plt.savefig(prefix + "_corr_learned.png")


