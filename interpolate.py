""" Code taken directly from https://github.com/dribnet/plat/blob/master/plat/interpolate.py """

import numpy as np
from scipy.stats import norm

def lerp(val, low, high):
    """Linear interpolation"""
    return low + (high - low) * val

def lerp_gaussian(val, low, high):
    """Linear interpolation with gaussian CDF"""
    low_gau = norm.cdf(low)
    high_gau = norm.cdf(high)
    lerped_gau = lerp(val, low_gau, high_gau)
    return norm.ppf(lerped_gau)

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def slerp_gaussian(val, low, high):
    """Spherical interpolation with gaussian CDF (generally not useful)"""
    offset = norm.cdf(np.zeros_like(low))  # offset is just [0.5, 0.5, ...]
    low_gau_shifted = norm.cdf(low) - offset
    high_gau_shifted = norm.cdf(high) - offset
    circle_lerped_gau = slerp(val, low_gau_shifted, high_gau_shifted)
    epsilon = 0.001
    clipped_sum = np.clip(circle_lerped_gau + offset, epsilon, 1.0 - epsilon)
    result = norm.ppf(clipped_sum)
    return result

# flat torus geodesics
# see https://arxiv.org/pdf/1212.6206.pdf Appendix A
# low and high are 2d shape=(batch_size, dim).
def lerp_toroidal(val, low, high):
    assert low.shape[1] % 2 == 0
    low_dir  = np.arctan2(low [:, 1::2], low [:, 0::2])
    high_dir = np.arctan2(high[:, 1::2], high[:, 0::2])
    # TODO need to go the shorter path out of the two possible
    interp_dir = low_dir + (high_dir - low_dir) * val
    result = np.zeros_like(low)
    result[:, 0::2] = np.cos(interp_dir)
    result[:, 1::2] = np.sin(interp_dir)
    return result

# low and high are 1d shape=(dim, ).
def lerp_toroidal_1d(val, low, high):
    assert len(low.shape)==len(high.shape)==1
    result = lerp_toroidal(val, low[np.newaxis, :], high[np.newaxis, :])
    return result[0]

def test_toroidal():
    import samplers

    z1 = samplers.toroidal_sampler(6, 4)
    z2 = samplers.toroidal_sampler(6, 4)

    if False:
        print(z1)
        print(z2)
        print(lerp_toroidal(0.0, z1, z2))
        print(lerp_toroidal(1.0, z1, z2))
        print(lerp_toroidal(0.1, z1, z2))
        print(lepr_toroidal(0.5, z1, z2))

    xs = []
    ys = []
    radius = 1.0
    for i in range(len(z1)):
        for val in np.linspace(0, 1, 20):
            xs.append(lerp_toroidal(val, z1, z2)[i, 0] * radius)
            ys.append(lerp_toroidal(val, z1, z2)[i, 1] * radius)
        radius -= 0.1
        if radius <= 0:
            break

    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.scatter(xs, ys)
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)
    plt.show()


def get_interpfn(spherical, gaussian, toroidal=False):
    """Returns an interpolation function"""
    if spherical and gaussian:
        return slerp_gaussian
    elif toroidal:
        return lerp_toroidal_1d
    elif spherical:
        return slerp
    elif gaussian:
        return lerp_gaussian
    else:
        return lerp


if __name__ == "__main__":
    test_toroidal()
