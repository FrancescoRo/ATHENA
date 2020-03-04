import numpy as np
from matplotlib import pyplot as plt
import GPy


def GPR(x, f, n_train=None, kernel=None, plot=False):
    """Execute Gaussian processes regression for input data x and targets f.

    Parameters
    ---------
    x : ndarray
        vector of n-dimensional samples
    f : ndarray
        vector of targets
    n_train : int, optional
        number of data in the training set, the rest is used for the test set
    kernel : kernel instance, optional
        kernel for the gaussian process
    plot : _bool, optional
        switch for plotting the Gaussian process regression when possible

    Returns
    -------
    gp : gaussian process instance
        gaussian process instance from the library GPy
    """
    M, m = np.shape(x)[0], np.shape(x)[1]

    if kernel is None:
        kernel = GPy.kern.RBF(input_dim=m, ARD=True)

    if n_train is not None:
        X = x[:n_train].reshape(-1, m)
        Y = f[:n_train].reshape(-1, 1)

        gp = GPy.models.GPRegression(X, Y, kernel)
        gp.optimize()
        Y_test = (gp.predict((x[n_train:]).reshape(-1, m)))[0]
        F_test = f[n_train:].reshape(-1, 1)

        #evaluate Relative Root Mean Squared Error
        RRMSE = np.sqrt(np.sum((Y_test-F_test)**2))/\
                        np.sqrt(np.sum((F_test-F_test.mean())**2))

        if plot is True:
            gp.plot()
            plt.scatter(x[n_train:], F_test, c=F_test)
            plt.show()

        return RRMSE, gp
    else:
        if len(f.shape) == 1:
            f = f.reshape(-1, 1)
        gp = GPy.models.GPRegression(x, f, kernel)
        gp.optimize()

        if plot is True:
            gp.plot()
            plt.show()

        return gp
