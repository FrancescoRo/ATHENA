"""[summary]

[description]
"""
import numpy as np
import matplotlib.pyplot as plt
from .subspaces import Subspaces
from .utils import (Normalizer, initialize_weights, sort_eigpairs)
from .feature_map import (FeatureMap, ProjectionMap)
from .tools import clock
from scipy import optimize

DEBUG = False


class NonlinearActiveSubspaces(Subspaces):
    """Nonlinear Active Subspaces class

    [description]
    """
    def __init__(self):
        super().__init__()
        self.n_features = None
        self.feature_map = None
        self.features = None
        self.pseudo_gradients = None

    @staticmethod
    @clock(activate=DEBUG)
    def _build_decompose_cov_matrix(inputs=None,
                                    outputs=None,
                                    gradients=None,
                                    weights=None,
                                    method=None):

        if method == 'exact' or method == 'local':
            cov_matrix = gradients.T.dot(gradients * weights)
            evals, evects = sort_eigpairs(cov_matrix)

        return cov_matrix, evals, evects

    def forward(self, inputs):
        """
        Map full variables to active and inactive variables.
        
        Points in the original input space are mapped to the active and
        inactive subspace.
        
        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active
            variables; array n_samples-by-inactive_dim containing the mapped
            inactive variables.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        features = np.array([
            self.feature_map.map(inputs[i, :]) for i in range(inputs.shape[0])
        ])
        active = np.dot(features, self.W1)
        inactive = np.dot(features, self.W2)
        return active, inactive

    @staticmethod
    @clock(activate=DEBUG)
    def _reparametrize(inputs=None,
                       outputs=None,
                       gradients=None,
                       n_features=None,
                       feature_map=None,
                       hyperparams=None):

        M, m = inputs.shape
        feature_map.compute(hyperparams)

        #Initialize Jacobian for each input
        jacobian = np.array(
            [feature_map.jacobian(inputs[i, :]) for i in range(M)])
        #Compute pseudo gradients
        pseudo_gradients = np.array([
            np.linalg.lstsq(jacobian[i, :, :].T, gradients[i, :].T,
                            rcond=None)[0] for i in range(M)
        ])
        #Compute features
        features = np.array([feature_map.map(inputs[i, :]) for i in range(M)])
        return pseudo_gradients, features

    def compute(self,
                hyperparams=None,
                inputs=None,
                outputs=None,
                gradients=None,
                weights=None,
                method='exact',
                nboot=None,
                n_features=None,
                feature_map=None):
        """[summary]

        Parameters
        ----------
        gradients : ndarray
            M-by-m matrix containing the gradient samples oriented as rows
        weights : ndarray
            M-by-1 weight vector, corresponds to numerical quadrature rule
            used to estimate matrix whose eigenspaces define the active
            subspace.

        Local linear models: This approach is related to the sufficient
        dimension reduction method known sometimes as the outer product
        of gradient method. See the 2001 paper 'Structure adaptive approach
        for dimension reduction' from Hristache, et al.
        """
        if method == 'exact':
            if gradients is None or inputs is None:
                raise ValueError('gradients or inputs argument is None.')
            if weights is None:
                # default weights is for Monte Carlo
                weights = initialize_weights(gradients)

            if n_features is None:
                self.n_features = inputs.shape[1]
            else:
                self.n_features = n_features

            if feature_map is None:
                #default spectral measure is Gaussian
                raise ValueError('feature_map argument is None.')
            else:
                self.feature_map = feature_map

            self.pseudo_gradients, self.features = self._reparametrize(
                inputs, outputs, gradients, self.n_features, self.feature_map,
                hyperparams)

            self.cov_matrix, self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=self.pseudo_gradients, weights=weights, method=method)

            if nboot is not None:
                self._compute_bootstrap_ranges(psuedo_gradients,
                                               weights,
                                               method=method,
                                               nboot=nboot)

        # estimate active subspace with local linear models.
        if method == 'local':
            if inputs is None or outputs is None:
                raise ValueError('inputs or outputs argument is None.')
            gradients = local_linear_gradients(inputs=inputs,
                                               outputs=outputs,
                                               weights=weights)
            if weights is None:
                # use the new gradients to compute the weights,
                # otherwise dimension mismatch accours.
                weights = initialize_weights(gradients)

            if n_features is None:
                self.n_features = inputs.shape[1]
            else:
                self.n_features = n_features

            if feature_map is None:
                #default spectral measure is Gaussian
                raise ValueError('feature_map argument is None.')
            else:
                self.feature_map = feature_map

            self.pseudo_gradients, self.features = self._reparametrize(
                inputs, gradients, self.n_features, self.feature_map,
                *hyperparams)

            self.cov_matrix, self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=self.pseudo_gradients, weights=weights, method=method)

            if nboot is not None:
                self._compute_bootstrap_ranges(psuedo_gradients,
                                               weights,
                                               method=method,
                                               nboot=nboot)


def tune(ranges=None, plot=None, opt='brute', **kw):
    """Optimize the parameters of the given distribution,
       with respect to the RRMSE. The parameters are optimized with
       logarithmic grid-search.
    """

    #List that collects the parameters evaluted in the optimization in the
    #first component and the corresponding RRMSE in the second.
    data = [[], []]
    n_params = len(ranges)

    #function to optimize
    fun = lambda hyperparams: Average_NRMSE(10**hyperparams, data, **kw)

    #TODO: provisional design
    if opt == 'brute':
        #logarithmic grid search
        res, val, grid, eval_grid = optimize.brute(fun,
                                                   ranges,
                                                   finish=None,
                                                   full_output=True)
        res = 10**res
    elif opt == 'differential_evolution':
        opt_res = optimize.differential_evolution(func,
                                                  ranges,
                                                  maxiter=30,
                                                  disp=True,
                                                  tol=0.01)
        res = 10**opt_res.x
        val = opt_res.fun
    elif opt == 'shgo':
        opt_res = optimize.shgo(fun, ranges)
        res = 10**opt_res.x
        val = opt_res.fun
    elif opt == 'dual_annealing':
        opt_res = optimize.dual_annealing(fun,
                                          ranges,
                                          maxiter=30,
                                          no_local_search=True)
        res = 10**opt_res.x
        val = opt_res.fun

    #plot to show dependance of NRMSE from the parameters
    if plot is True:
        # plt.plot(grid, eval_grid)
        # plt.grid(True, linestyle='dotted')
        input = np.log10(np.array(data[0][1:]))
        output = np.array(data[1][1:])

        if n_params == 1:
            plt.plot(input, output)
            plt.grid(True, linestyle='dotted')
            plt.xlabel("parameter")
            plt.ylabel("NRMSE")

        elif n_params == 2:
            fig = plt.figure()
            plt.tricontourf(input[:, 0], input[:, 1], output, 15)
            plt.colorbar()
            plt.xlabel("first_hyperparameter")
            plt.ylabel("second_hyperparameter")

        elif n_params == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111,
                                 projection='3d',
                                 title="Best {}".format(res))
            ax.scatter(input[:, 1], input[:, 2], output)
        plt.show()
    return res, val


from .cross_validation import Estimator, cross_validation


@clock(activate=DEBUG, fmt='[{elapsed:0.8f}s] {name}\n')
def Average_NRMSE(hyperparams, data, inputs, outputs, gradients, n_features,
                  feature_map, weights, method, kernel, gp_dimension, folds):
    """Function to optimize."""

    GPR_NAS = Estimator(hyperparams, 'NAS', n_features, feature_map, weights,
                        method, kernel, gp_dimension)

    mean, std = cross_validation(inputs, outputs, gradients, GPR_NAS, folds)

    print("params {2} mean {0}, std {1}".format(mean, std, hyperparams))
    data[0].append(hyperparams)
    data[1].append(mean)

    return mean
