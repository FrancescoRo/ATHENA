"""[summary]

[description]
"""
import numpy as np
from scipy.stats import cauchy


class FeatureMap():
    """Feature map

    [description]
    """
    def __init__(fm):
        fm._map = None
        fm._jacobian = None
        fm.n_features = None
        fm.input_dim = None

    def compute(fm,
                map=None,
                jacobian=None,
                n_features=None,
                input_dim=None,
                **hyperparams):
        pass

    def map(fm, x):
        pass

    def jacobian(fm, x):
        pass


class ProjectionMap(FeatureMap):
    def __init__(pr, feature_map, feature_map_jac, distr, n_params, input_dim,
                 n_features, **kw):
        super().__init__()
        pr.params = None
        pr.matrix = None
        pr._map = feature_map
        pr._jacobian = feature_map_jac
        pr.distr = distr
        pr.input_dim = input_dim
        pr.n_features = n_features
        pr.n_params = n_params
        pr.kw = kw

    def map(pr, x):
        return pr._map(x,
                       pr.matrix,
                       pr.params,
                       n_features=pr.n_features,
                       n_params=pr.n_params,
                       input_dim=pr.input_dim,
                       **pr.kw)

    def jacobian(pr, x):
        return pr._jacobian(x,
                            pr.matrix,
                            pr.params,
                            n_features=pr.n_features,
                            n_params=pr.n_params,
                            input_dim=pr.input_dim,
                            **pr.kw)

    def compute(pr, params):
        pr.params = params
        pr.matrix = pr.set_projection_matrix(pr.distr, pr.input_dim,
                                             pr.n_features, pr.n_params, params)

    @staticmethod
    def set_projection_matrix(distr, m, n_features, n_params, params):
        """Set the projection matrix and the hyperparameter sigma_f of the feature
           map. Several spectral measures can be chosen."""

        if distr == np.random.multivariate_normal:
            pr_matrix = distr(np.zeros(m), params[0] * np.diag(np.ones(m)),
                              (n_features))
        elif distr == np.random.dirichlet:
            pr_matrix = distr(params[0] * np.ones(m), (n_features))
        elif distr == cauchy:
            pr_matrix = distr.rvs(0, params[0], (n_features, m))
        elif distr == np.random.laplace:
            pr_matrix = distr(0, params[0], (n_features, m))
        elif distr == np.random.uniform:
            pr_matrix = distr(params[0], params[1], (n_features, m))
        elif distr == np.random.beta:
            pr_matrix = distr(params[0], params[1], (n_features, m))

        return pr_matrix


def RFF_map(x, W, params, **kw):
    return np.sqrt(
        2 / kw['n_features']) * kw['sigma_f'] * np.cos(W.dot(x) + kw['b'])


def RFF_jac(x, W, params, **kw):
    return Hadamard(
        np.sqrt(2 / kw['n_features']) * kw['sigma_f'] * (-1) *
        np.sin(W.dot(x) + kw['b']), W, **kw)


def Hadamard(M, W, **kw):  #(nfeatures, (nfeatures, m))
    return M.reshape(kw['n_features'], 1) * W.reshape(kw['n_features'],
                                                      kw['input_dim'])


def atan_map(x, W, params, **kw):
    return kw['sigma_f'] * np.arctan(W.dot(x) + kw['b'])


def atan_jac(x, W, params, **kw):
    return Hadamard(kw['sigma_f'] / (1 + (W.dot(x) + kw['b'])**2), W, **kw)
