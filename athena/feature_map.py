"""FeatureMap class and ProjectionMap subclass
[description]
"""
import numpy as np
from scipy.stats import cauchy, dirichlet


class FeatureMap(object):
    """doc"""
    def __init__(self, feature_map, feature_map_jac, distr, n_params, input_dim,
                 n_features, **kw):
        super().__init__()
        self.params = None
        self.matrix = None
        self._fmap = feature_map
        self._jacobian = feature_map_jac
        self.distr = distr
        self.input_dim = input_dim
        self.n_features = n_features
        self.n_params = n_params
        self.kw = kw
        self._best_matrix = None
        self._best_params = None
        self.score = None
        self.tuned = False

    def fmap(self, x):
        """doc"""
        if self.tuned:
            return self._fmap(x,
                              self._best_matrix,
                              self._best_params,
                              n_features=self.n_features,
                              n_params=self.n_params,
                              input_dim=self.input_dim,
                              **self.kw)
        else:
            return self._fmap(x,
                              self.matrix,
                              self.params,
                              n_features=self.n_features,
                              n_params=self.n_params,
                              input_dim=self.input_dim,
                              **self.kw)

    def jacobian(self, x):
        if self.tuned:
            return self._jacobian(x,
                                  self._best_matrix,
                                  self._best_params,
                                  n_features=self.n_features,
                                  n_params=self.n_params,
                                  input_dim=self.input_dim,
                                  **self.kw)
        else:
            return self._jacobian(x,
                                  self.matrix,
                                  self.params,
                                  n_features=self.n_features,
                                  n_params=self.n_params,
                                  input_dim=self.input_dim,
                                  **self.kw)

    def compute(self, params):
        if self.tuned is False:
            self.params = params
            self.matrix = self.set_projection_matrix(self.distr, self.input_dim,
                                                     self.n_features,
                                                     self.n_params, params)
    
    def set_params(self, **kw):
        self.matrix = kw['matrix']

    def set_best(self, score):
        if self.score is None:
            self.score = score
            self._best_matrix = self.matrix
            self._best_params = self.params
        elif self.score > score:
            self.score = score
            self._best_matrix = self.matrix
            self._best_params = self.params

    def set_tuned(self):
        self.tuned = True

    def get_best(self):
        return self._best_matrix, self._best_params

    @staticmethod
    def set_projection_matrix(distr, m, n_features, n_params, params):
        """Set the selfojection matrix and the hyperparameter sigma_f of the feature
           map. Several spectral measures can be chosen."""

        if distr == np.random.multivariate_normal:
            pr_matrix = distr(np.zeros(m), np.diag(params),
                              (n_features))
        elif distr == np.random.normal:
            pr_matrix = distr(0, params[0], (n_features, m))
        elif distr == np.random.dirichlet:
            pr_matrix = distr(params[0] * np.ones(m), (n_features))
        elif distr == 'cauchy':
            pr_matrix = cauchy.rvs(params, size=n_features)
        elif distr == np.random.laplace:
            pr_matrix = distr(params[0], params[1], (n_features, m))
        elif distr == np.random.uniform:
            pr_matrix = distr(params[0], params[1], (n_features, m))
        elif distr == np.random.beta:
            pr_matrix = distr(params[0], params[1], (n_features, m))
        elif distr == 'dirichlet':
            pr_matrix = dirichlet.rvs(params, size=n_features)
        else:
            pr_matrix = distr(params,
                              m=m,
                              n_features=n_features,
                              n_params=n_params)

        return pr_matrix


def Hadamard(M, W, **kw):  # (nfeatures, (nfeatures, m))
    """doc"""
    return M.reshape(kw['n_features'], 1) * W.reshape(kw['n_features'],
                                                      kw['input_dim'])


def RFF_map(x, W, params, **kw):
    """doc"""
    return np.sqrt(
        2 / kw['n_features']) * kw['sigma_f'] * np.cos(W.dot(x) + kw['b'])


def RFF_jac(x, W, params, **kw):
    """doc"""
    return Hadamard(
        np.sqrt(2 / kw['n_features']) * kw['sigma_f'] * (-1) *
        np.sin(W.dot(x) + kw['b']), W, **kw)