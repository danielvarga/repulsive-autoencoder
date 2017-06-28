import numpy as np

def gaussian_sampler(batch_size, latent_dim):
    return np.random.normal(size=(batch_size, latent_dim))
def spherical_sampler(batch_size, latent_dim):
    z_sample = np.random.normal(size=(batch_size, latent_dim))
    z_sample /= np.linalg.norm(z_sample, axis=1, keepdims=True)
    return z_sample

def sampler_factory(args, x_train):
    if args.decoder == "gaussianxxx":
        def train_sampler(batch_size, latent_dim):
            return x_train[:batch_size]
        return train_sampler
    elif args.spherical:
        return spherical_sampler
    else:
        return gaussian_sampler

# TODO move samplers from project_latent.py here
