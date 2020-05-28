"""Gaussian Process Regression (GPR) from GPy
[description]
"""

import numpy as np
from matplotlib import pyplot as plt
import GPy


def gpr(x, f, n_train=None, kernel=None, plot=False):
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
    m = x.shape[1]
    
    if kernel is None:
        kernel = GPy.kern.RBF(input_dim=m, ARD=True)

    gp = GPy.models.GPRegression(x, f.reshape(-1, 1), kernel)
    gp.optimize()

    if plot:
        gp.plot()
        plt.show()

    return gp
