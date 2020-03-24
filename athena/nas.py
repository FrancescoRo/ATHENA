"""[summary]

[description]
"""
import numpy as np
from .subspaces import Subspaces
from .utils import (initialize_weights, sort_eigpairs, local_linear_gradients)
from .tools import clock

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
    def _build_decompose_cov_matrix(gradients=None, weights=None, method=None):

        if method == 'exact' or method == 'local':
            cov_matrix = gradients.T.dot(gradients * weights)
            evals, evects = sort_eigpairs(cov_matrix)

        return cov_matrix, evals, evects

    def forward(self, inputs):
        """
        Map full variables to active and inactive variables.
        Points in the original input space are mapped to the active and
        inactive non-linear subspace.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :return: array n_samples-by-active_dim containing the mapped active
            variables; array n_samples-by-inactive_dim containing the mapped
            inactive variables.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        features = np.array([
            self.feature_map.fmap(inputs[i, :]) for i in range(inputs.shape[0])
        ])
        active = np.dot(features, self.W1)
        inactive = np.dot(features, self.W2)
        return active, inactive

    def backward(self, reduced_inputs, n_points):
        pass

    @staticmethod
    @clock(activate=DEBUG)
    def _reparametrize(inputs=None, gradients=None, feature_map=None):

        M = inputs.shape[0]

        # Initialize Jacobian for each input
        jacobian = np.array(
            [feature_map.jacobian(inputs[i, :]) for i in range(M)])
        # Compute pseudo gradients
        pseudo_gradients = np.array([
            np.linalg.lstsq(jacobian[i, :, :].T, gradients[i, :].T,
                            rcond=None)[0] for i in range(M)
        ])
        # Compute features
        features = np.array([feature_map.fmap(inputs[i, :]) for i in range(M)])
        return pseudo_gradients, features

    def compute(self,
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
                # default spectral measure is Gaussian
                raise ValueError('feature_map argument is None.')
            else:
                self.feature_map = feature_map

            self.pseudo_gradients, self.features = self._reparametrize(
                inputs, gradients, self.feature_map)

            self.cov_matrix, self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=self.pseudo_gradients,
                weights=weights,
                method=method)

            if nboot:
                self._compute_bootstrap_ranges(self.psuedo_gradients,
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
                # default spectral measure is Gaussian
                raise ValueError('feature_map argument is None.')
            else:
                self.feature_map = feature_map

            self.pseudo_gradients, self.features = self._reparametrize(
                inputs, gradients, self.feature_map)

            self.cov_matrix, self.evals, self.evects = self._build_decompose_cov_matrix(
                gradients=self.pseudo_gradients,
                weights=weights,
                method=method)

            if nboot:
                self._compute_bootstrap_ranges(self.psuedo_gradients,
                                               weights,
                                               method=method,
                                               nboot=nboot)