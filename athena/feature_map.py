"""FeatureMap class and ProjectionMap subclass
[description]
"""
import numpy as np
from scipy.stats import cauchy, dirichlet


class FeatureMap():
    """Feature map

    [description]
    """
    def __init__(self):
        self._fmap = None
        self._jacobian = None
        self.n_features = None
        self.input_dim = None

    def compute(self,
                fmap=None,
                jacobian=None,
                n_features=None,
                input_dim=None,
                **hyperparams):
        """doc"""

    def fmap(self, x):
        """doc"""

    def jacobian(self, x):
        """doc"""


class ProjectionMap(FeatureMap):
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
        self.score = 0
        self._tuned = False

    def fmap(self, x):
        """doc"""
        if self._tuned:
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
        if self._tuned:
            print(self._best_matrix)
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
        if self._tuned is False:
            self.params = params
            self.matrix = self.set_projection_matrix(self.distr, self.input_dim,
                                                     self.n_features,
                                                     self.n_params, params)

    def set_best(self, score):
        self.score += 1
        #self.score = score
        self._best_matrix = self.matrix
        self._best_params = self.params

    def set_tuned(self):
        self._tuned = True

    @staticmethod
    def set_projection_matrix(distr, m, n_features, n_params, params):
        """Set the selfojection matrix and the hyperparameter sigma_f of the feature
           map. Several spectral measures can be chosen."""

        if distr == np.random.multivariate_normal:
            pr_matrix = distr(np.zeros(m), params * np.diag(np.ones(m)),
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


def atan_map(x, W, params, **kw):
    """doc"""
    return kw['sigma_f'] * np.arctan(W.dot(x) + kw['b'])


def atan_jac(x, W, params, **kw):
    """doc"""
    return Hadamard(kw['sigma_f'] / (1 + (W.dot(x) + kw['b'])**2), W, **kw)


def asin_map(x, W, params, **kw):
    """doc"""
    return kw['sigma_f'] * np.arcsin(W.dot(x) + kw['b'])


def asin_jac(x, W, params, **kw):
    """doc"""
    return Hadamard((-1) * kw['sigma_f'] / np.sqrt(1 - (W.dot(x) + kw['b'])**2),
                    W, **kw)
