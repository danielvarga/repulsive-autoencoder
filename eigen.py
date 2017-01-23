import numpy as np
import keras
import keras.backend as K

def eigvec_of_cov(W, n, latent_dim, iterations=9, inner_normalization=False):
    WW = K.dot(K.transpose(W), W)
    return eigvec(WW, n, latent_dim=latent_dim, iterations=iterations, inner_normalization=inner_normalization)


def extreme_eigvals(WW, n, latent_dim, iterations=9, inner_normalization=False):
    domineigvec, domineigval = eigvec(WW, n, latent_dim=latent_dim, iterations=iterations, inner_normalization=inner_normalization)
    WW_shifted = K.eye(latent_dim) * domineigval - WW
    domineigvec2, domineigval2 = eigvec(WW_shifted, n, latent_dim=latent_dim, iterations=iterations, inner_normalization=inner_normalization)
    domineigval2 += domineigval
    return domineigval2, domineigval


# POWER METHOD FOR APPROXIMATING THE DOMINANT EIGENVECTOR OF SYMMETRIC POSITIVE DEFINITE MATRIX:
# TODO The dependence on n is ridiculous, get rid of it.
def eigvec(WW, n, latent_dim, iterations=9, inner_normalization=False):
    o = K.ones([latent_dim, 1]) # initial values for the dominant eigenvector
    domineigvec = o
    for i in range(iterations):
        domineigvec = K.dot(WW, domineigvec)
        if inner_normalization:
            domineigvec = domineigvec / K.sqrt(K.dot(K.transpose(domineigvec), domineigvec))
    if not inner_normalization:
        domineigvec = domineigvec / K.sqrt(K.dot(K.transpose(domineigvec), domineigvec))
    WWd = K.transpose(K.dot(WW, domineigvec))
    domineigval = K.dot(WWd, domineigvec) / n # THE CORRESPONDING DOMINANT EIGENVALUE
    return domineigvec, domineigval


def test_eigenvec():
    bs = 200
    latent_dim = 20

    input = K.placeholder(shape=(bs, latent_dim))

    data = np.random.normal(size=(bs, latent_dim))
    # putting some correlation in there:
    data[:, 0] += 2 * data[:, 1]

    print data.shape
    cov = np.cov(data.T)
    print "empirical cov", cov

    # This corresponds to the above specific data[:, 0] += 2 * data[:, 1]
    theo_cov = np.eye(latent_dim)
    theo_cov[0, 0] = 5
    theo_cov[1, 0] = 2
    theo_cov[0, 1] = 2

    eigVals, eigVects = np.linalg.eigh(cov)
    print "eigvals = ", list(reversed(eigVals))
    print "dominant eigvect = ", eigVects[:, -1]

    theo_eigVals, theo_eigVects = np.linalg.eigh(theo_cov)
    print "theoretical eigvals = ", list(reversed(theo_eigVals))
    print "theoretical dominant eigvect = ", theo_eigVects[:, -1]

    print "======="

    f = K.function([input], list(eigvec_of_cov(input, bs, latent_dim, iterations=3)))
    domEigVect, domEigVal = f([data])
    print domEigVect.shape, domEigVal.shape
    print "iterative keras-based dominant eigenvalue", domEigVal[0][0]
    print "iterative keras-based dominant eigenvector", domEigVect[:, 0]

    print "======="

    WW = K.dot(K.transpose(input), input)
    f = K.function([input], list(extreme_eigvals(WW, bs, latent_dim, iterations=3)))
    mineigval, maxeigval = f([data])
    print "iterative keras-based extreme eigenvalues", mineigval, maxeigval

    theo_cov_shifted = np.eye(latent_dim) * theo_eigVals[0] - theo_cov
    eigVals, eigVects = np.linalg.eigh(theo_cov_shifted)
    # print "eigvals shifted", eigVals
    # print "smallest theoretical eigval = ", eigVals[-1] + theo_eigVals[0]


if __name__ == "__main__":
    test_eigenvec()
