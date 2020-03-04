import numpy as np
from scipy import optimize
from scipy.stats import beta
from scipy.stats import cauchy
import matplotlib.pyplot as plt

def radial(r, x, normalizer):
    #each row of xx should be  in the normalized space

    M, m = x.shape

    x = normalizer.unnormalize(x)

    f = np.array([r(np.dot(x[i].T, x[i])) for i in range(M)])

    return f

def radial_grad(dr, x, normalizer):
    #each row of xx should be in the normalized space

    M, m = x.shape

    x = normalizer.unnormalize(x)

    f = np.array([dr(np.dot(x[i].T, x[i])) for i in range(M)]).reshape(M, 1)

    return 2 * x * f

def norm(x):
    return np.sqrt(x)

def dnorm(x):
    return 0.5/np.sqrt(x)

def paraboloid(x):
    return x

def dparaboloid(x):
    return 1

def sin(x):
    return np.sin(x)

def dsin(x):
    return -np.cos(x)

def root(x):
    return x**3

def droot(x):
    return 3*x**2

def log(x):
    return np.log(x)

def dlog(x):
    return 1/x

def GPR_pretty_print(X, f, df, W, sigma_f, nfeatures, n_train, title, kernel=None):
    """Print gaussian process regression of NAS with given hyperparameters
       associated to the distribution used to compute y with NAS"""

    M, m = X.shape[0], X.shape[1]

    if n_train!=0:
        XX = X[:n_train, :]
        X_test = X[n_train:, :]

        ss = ac.subspaces.Subspaces()
        ss.compute(X=XX, f=f, df=df, sstype='NAS', nfeatures=nfeatures, pr_matrix=W, sigma_f=sigma_f)
        ss.partition(2)
        X_train = ss.Z.dot(ss.W1)

        Z = ss.feature_map(X_test)
        Y = Z.dot(ss.W1)
        y = (Y[:,0]).reshape(-1, 1)
        ac.utils.plotters.sufficient_summary_3D(X_train, f[:n_train].reshape((-1,)))
        ac.utils.plotters.eigenvalues(ss.eigenvals[:10])

        gp_NAS = ac.utils.gaussian_processes.GPRy(X_train[:, 0].reshape(-1,1), f[:n_train], kernel=kernel)
        gp_NAS.plot(title='{} NAS'.format(title))
        plt.scatter(y[:, 0], f[n_train:, 0], c=f[n_train:, 0])
        plt.xlabel('non-linear active subspace')
        plt.ylabel('model f')
        plt.grid(True)
        plt.show()

    else:
        XX = X

        ss = ac.subspaces.Subspaces()
        ss.compute(X=XX, f=f, df=df, sstype='NAS', nfeatures=nfeatures, pr_matrix=W, sigma_f=sigma_f)
        ss.partition(2)
        X_train = (ss.Z.dot(ss.W1)[:, 0]).reshape(-1, 1)

        gp_NAS = ac.utils.gaussian_processes.GPRy(X_train[:, 0], f[:n_train], kernel=kernel)
        gp_NAS.plot(title='{} NAS'.format(title))
        plt.xlabel('active subspace')
        plt.ylabel('f')
        plt.grid(True)
        plt.show()
