import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = "/home/csadrian/improved_wgan_training/gradient_generated_5999.npy"
    data = np.load(filename).reshape((-1, 64, 64, 3))
    print data.shape
    data = data.mean(axis=3)
    print data.shape
    mask = np.zeros((1, 64, 64, 2))
    for y in range(64):
        for x in range(64):
            yy = 2 * float(y) / 64 - 1
            xx = 2 * float(x) / 64 - 1
            mask[0, y, x, 0] = yy
            mask[0, y, x, 1] = xx
    prod = mask * data.reshape((-1, 64, 64, 1))
    print prod.shape
    grads = prod.mean(axis=(1, 2))
    print grads.shape
    centers = grads.mean(axis=0)
    print centers
    grads -= centers
    plt.hexbin(grads[:, 0], grads[:, 1], gridsize=50, cmap='inferno')
    plt.savefig("gradscatter.png")
    plt.close()
    dirs = np.arctan2(grads[:, 1], grads[:, 0])
    plt.hist(dirs, 180, normed=1)
    plt.savefig("dirhist.png")

main()
