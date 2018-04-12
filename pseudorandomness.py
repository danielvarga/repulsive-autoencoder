import numpy as np


def main_effect_of_sampling():
    N = 5000
    n = 500
    A = np.random.normal(size=(N, n))

    cov = np.cov(A.T)
    print(cov.shape)
    eigvals = list(np.linalg.eigvals(cov).real)
    print("cov eigvals = ", sorted(eigvals, reverse=True))
    import matplotlib.pyplot as plt
    plt.hist(eigvals, bins=30)
    plt.savefig("simulated_eigs_N%d_n%d.png" % (N, n))
    # cho = np.linalg.cholesky(cov)


def main_pseudorandomness():
    N = 200
    n = 20
    m = 10

    indexes = np.array([np.random.choice(n, size=m, replace=False) for _ in range(N)])

    cov = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            i1 = indexes[p]
            i2 = indexes[q]
            inters = len(set(i1) & set(i2))
            cov[p][q] = (float(inters) / m) ** 2

    eigvals = list(np.linalg.eigvals(cov).real)
    print("cov eigvals = ", sorted(eigvals, reverse=True))
    import matplotlib.pyplot as plt
    plt.hist(eigvals, bins=30)
    plt.savefig("pseudorandom_eigs_N%d_n%d_m%d.png" % (N, n, m))


main_pseudorandomness()
