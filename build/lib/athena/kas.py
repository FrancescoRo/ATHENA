"""[summary]

[description]
"""
import numpy as np
from .subspaces import Subspaces
from .utils import (initialize_weights, sort_eigpairs, local_linear_gradients)
import matplotlib.pyplot as plt

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
    def _build_decompose_cov_matrix(pseudo_gradients=None,
                                    weights=None,
                                    method=None,
                                    metric=None,
                                    input_cov=None):
        """
        Computes the uncentered covariance matrix of the pseudo_gradients.

        :param numpy.ndarray pseudo_gradients: array n_samples-by-n_features containing
            the psuedo gradients.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds to numerical
            quadrature rule used to estimate matrix whose eigenspaces define the active
            subspace.
        :param str method: the method used to compute the gradients.
        :return: array n_features-by-n_features representing the uncentered covariance matrix;
            array n_features containing the eigenvalues ordered decreasingly in magnitude;
            array n_features-by-n_features the columns contain the ordered eigenvectors.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        if method == 'exact' or method == 'local':
            cov_matrix = np.array(np.sum([weights[i, 0]*np.dot(pseudo_gradients[i, :, :].T, np.dot(metric, pseudo_gradients[i,:,:])) for i in range(pseudo_gradients.shape[0])], axis=0))
            evals, evects = sort_eigpairs(cov_matrix, input_cov)
        return evals, evects
            # X = np.squeeze(pseudo_gradients * np.sqrt(weights).reshape(-1, 1, 1))
            # _, singular, evects = np.linalg.svd(X, full_matrices=False)
            # evals = singular**2
        # return evals, evects.T

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
    def _reparametrize(inputs=None, gradients=None, feature_map=None):
        """
        Computes the pseudo-gradients solving an overdetermined linear system.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :param numpy.ndarray gradients: array n_samples-by-n_params containing
            the gradient samples oriented as rows.
        :param feature_map: feature map object.
        :return: array n_samples-by-n_features containing
            the psuedo gradients; array n_samples-by-n_features containing the
            image of the inputs in the feature space.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        n_samples = inputs.shape[0]
        # Initialize Jacobian for each input
        jacobian = np.array(
            [feature_map.jacobian(inputs[i, :]) for i in range(n_samples)])
        
        # Compute pseudo gradients
        pseudo_gradients = np.array([
            np.linalg.lstsq(jacobian[i, :, :].T,
                            gradients[i, :, :].T,
                            rcond=None)[0].T for i in range(n_samples)
        ])
        # pseudo_gradients = np.array([np.dot(gradients[i, :, :], np.linalg.pinv(jacobian[i, :, :])) for i in range(n_samples)])

        # Compute features
        features = np.array(
            [feature_map.fmap(inputs[i, :]) for i in range(n_samples)])
        return pseudo_gradients, features

    def compute(self,
                inputs=None,
                outputs=None,
                gradients=None,
                weights=None,
                method='exact',
                nboot=None,
                n_features=None,
                feature_map=None,
                metric=None,
                input_cov=None):
        """
        [Description]
        Local linear models: This approach is related to the sufficient
        dimension reduction method known sometimes as the outer product
        of gradient method. See the 2001 paper 'Structure adaptive approach
        for dimension reduction' from Hristache, et al.

        :param numpy.ndarray inputs: array n_samples-by-n_params containing
            the points in the original parameter space.
        :param numpy.ndarray outputs: array n_samples-by-1 containing
            the values of the model function.
        :param numpy.ndarray gradients: array n_samples-by-n_params containing
            the gradient samples oriented as rows.
        :param numpy.ndarray weights: n_samples-by-1 weight vector, corresponds to numerical
            quadrature rule used to estimate matrix whose eigenspaces define the active
            subspace.
        :param str method: the method used to compute the gradients.
        :param int nboot: number of bootstrap samples.
        :param int n_features: dimension of the feature space.
        :param feature_map: feature map object.
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

            if metric is None:
                self.metric = np.eye(gradients.shape[1])
            else:
                self.metric = metric

            if input_cov is None:
                self.input_cov = np.identity(n_features)
            else:
                self.input_cov = input_cov

            self.pseudo_gradients, self.features = self._reparametrize(
                inputs, gradients, self.feature_map)

            self.evals, self.evects = self._build_decompose_cov_matrix(
                pseudo_gradients=self.pseudo_gradients,
                weights=weights,
                method=method,
                metric=self.metric, input_cov=self.input_cov)

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
                pseudo_gradients=self.pseudo_gradients,
                weights=weights,
                method=method)
            
            if nboot:
                self._compute_bootstrap_ranges(self.psuedo_gradients,
                                               weights,
                                               method=method,
                                               nboot=nboot)

    def plot_eigenvalues(self, num=10, filename=None, figsize=(8, 8), title=''):
        """
        Plot the eigenvalues.
        
        :param str filename: if specified, the plot is saved at `filename`.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        :raises: ValueError

        .. warning::
            `self.compute` has to be called in advance.
        """
        if self.evals is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the compute method.')
        n_pars = num
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.semilogy(range(1, n_pars + 1),
                     self.evals[:n_pars],
                     'ko-',
                     markersize=8,
                     linewidth=2)
        plt.xticks(range(1, n_pars + 1))
        plt.xlabel('Index')
        plt.ylabel('Eigenvalues')
        plt.grid(linestyle='dotted')
        if self.evals_br is None:
            plt.axis([
                0, n_pars + 1, 0.1 * np.amin(self.evals),
                10 * np.amax(self.evals)
            ])
        else:
            plt.fill_between(range(1, n_pars + 1),
                             self.evals_br[:, 0],
                             self.evals_br[:, 1],
                             facecolor='0.7',
                             interpolate=True)
            plt.axis([
                0, n_pars + 1, 0.1 * np.amin(self.evals_br[:, 0]),
                10 * np.amax(self.evals_br[:, 1])
            ])

        if filename:
            plt.savefig(filename)
        else:
            plt.show()