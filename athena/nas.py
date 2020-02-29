"""[summary]

[description]
"""
import numpy as np
from .subspaces import Subspaces
from .utils import (Normalizer, initialize_weights, sort_eigpairs)
from .feature_map import (Feature_map, Random_Fourier_feature_map)


class NonlinearActiveSubspaces(Subspaces):
    """Nonlinear Active Subspaces class

    [description]
    """

    def __init__(self):
        super().__init__()
        self.n_features = None
        self.features = None
        self.pseudo_gradients = None

    def compute(self,
                inputs=None,
                outputs=None,
                gradients=None,
                n_features=None,
                feature_map=None,
                **hyperparams):
        """[summary]
        Map the points in the active variable space to the original parameter
        space.

        :param numpy.ndarray inputs:
        :param numpy.ndarray outputs:
        :param numpy.ndarray gradients:
        :param int n_features:
        :param feature_map:
        :type: Feature_map
        :params Keyword Arguments hyperparameters:

        :return: n_samples-by-n_features matrix
                 n_samples-by-n_features matrix
        :rtype: numpy.ndarray, numpy.ndarray

        Notes
        -----
        The inverse map depends critically on the `regularize_z` function.
        """
        if gradients is None:
            raise ValueError('gradients argument is None.')
        if weights is None:
            # default weights is for Monte Carlo
            weights = initialize_weights(gradients)
        if feature_map is None:
            #default spectral measure is Gaussian
            feature_map = Random_Fourier_feature_map()
        if n_features is None:
            self.n_features = inputs.shape[1]
        else:
            self.n_features = n_features

        M, m = gradients.shape

        #Initialize Jacobian for each input
        jacobian = [feature_map.feature_map_Jac(inputs[i, :], hyperparams)
                    for i in range(M)]

        #Compute pseudo gradients
         self.pseudo_gradients = [np.linalg.lstsq(jacobian[i,:,:].T,
                                 gradients[i,:].T, rcond=None)[0]
                                 for i in range(M)]

        #Compute features
        self.features = [feature_map.map(X[i, :], params) for i in range(M)]

    def tune(X, f, df, nfeatures, n_params, ranges, distr=None, folds=3, plot=False,
             kernel=None, gp_dimension=1):

    """Optimize the parameters of the given distribution,
       with respect to the RRMSE. The parameters are optimized with
       logarithmic grid-search."""

    M, m = X.shape[0], X.shape[1]

    #list that collects the parameters evaluted in the optimization in the
    #first component and the corresponding RRMSE in the second.
    data = [[],[]]

    #function to optimize
    if fm is not None:
        fun = lambda params: Average_RRMSE(10**params, data, X, f, df, nfeatures,distr,
                                           n_params, folds, kernel, gp_dimension, sstype,
                                           fm)
    else:
        fun = lambda params: Average_RRMSE(10**params, data, X, f, df, nfeatures,distr,
                                           n_params, folds, kernel, gp_dimension, sstype)

    #logarithmic grid search
    res = optimize.brute(fun, ranges, finish=None)
    res = 10**res

    #plot to show dependance of RRMSE form the parameters
    if plot is True:
        input = np.log10(np.array(data[0][1:]))
        output = np.array(data[1][1:])

        if n_params==1:
            plt.plot(input, output)
            plt.grid(True, linestyle='dotted')
            plt.xlabel("parameter")
            plt.ylabel("NRMSE")

        elif n_params==2:
            fig = plt.figure()
            plt.tricontourf(input[:, 0], input[:, 1], output, 15)
            plt.colorbar()
            plt.xlabel("first_hyperparameter")
            plt.ylabel("second_hyperparameter")

        elif n_params==3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d',
                                 title="Best {}".format(res))
            ax.scatter(input[:, 1], input[:, 2], output)


    plt.show()

    return res

def Average_RRMSE(params, data, X, f, df, nfeatures, distr, n_params, folds,
                  kernel=None, gp_dimension=1, sstype='NAS', fm=None):

    """Function to optimize."""

    M, m = X.shape

    if fm is None:
        #set the distribution, n_params is at max 3
        pr_matrix, sigma_f = set_projection_matrix(params, distr, m, nfeatures, n_params)

        RRMSE = np.zeros(folds)
        GPR_NAS = cross_validation.estimator(sstype, nfeatures, pr_matrix, sigma_f, kernel,
                                             gp_dimension)
        mean, std = cross_validation.Cross_validation(X, f, df, GPR_NAS, folds=folds)
        print("params {2} mean {0}, std {1}".format(mean, std, params))

        data[0].append(params)
        data[1].append(mean)

        return mean
    else:

        RRMSE = np.zeros(folds)
        GPR_NAS = cross_validation.estimator(sstype, nfeatures, params=params, gp_dimension=gp_dimension,
                                             fm=fm)
        mean, std = cross_validation.Cross_validation(X, f, df, GPR_NAS, folds=folds)
        print("params {2} mean {0}, std {1}".format(mean, std, params))

        data[0].append(params)
        data[1].append(mean)

        return mean

def set_projection_matrix(params, distr, m, nfeatures, n_params):
    """Set the projection matrix and the hyperparameter sigma_f of the feature
       map. Several spectral measures can be chosen."""
    sigma_f=None
    pr_matrix=None

    if distr==np.random.multivariate_normal:
        sigma_f=None
        pr_matrix = distr(np.zeros(m), params[0]*np.diag(np.ones(m)), (nfeatures))
    elif distr==np.random.dirichlet:
        sigma_f=None
        pr_matrix = distr(params[0]*np.ones(m), (nfeatures))
    elif distr==cauchy:
        sigma_f=None
        pr_matrix = distr.rvs(0, params[0], (nfeatures, m))
    elif distr==np.random.laplace:
        sigma_f=None
        pr_matrix = distr(0, params[0], (nfeatures, m))
    elif distr==np.random.uniform:
        sigma_f=params[0]
        pr_matrix = distr(params[1], params[2], (nfeatures,m))
    elif distr==np.random.beta:
        sigma_f=None
        pr_matrix = distr(params[0], params[1], (nfeatures,m))
    elif n_params==1:
        sigma_f=None
        pr_matrix = distr(params[0], (nfeatures, m))
    elif n_params==2:
        sigma_f=params[0]
        pr_matrix = distr(params[1], (nfeatures, m))
    elif n_params==3:
        sigma_f=params[0]
        pr_matrix = distr(params[1], params[2], (nfeatures, m))

    return pr_matrix, sigma_f
