"""
Module for the factory class for projection matrices
"""
import numpy as np
from scipy.stats import cauchy


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class ProjectionFactory(object):
    """
    Factory class that spawns projection matrices.

    :Example:

        >>> from athena import ProjectionFactory
        >>> import numpy as np
        >>> input_dim = 5
        >>> n_features = 100
        >>> params = [1.1, 2.9]
        >>> for pname in ProjectionFactory.projections:
        >>>     y = ProjectionFactory(pname)(input_dim, n_features, params)
    """
    def beta(input_dim, n_features, params):
        """
        TO DOC
        """
        return np.random.beta(params[0], params[1], (n_features, input_dim))

    def cauchy(n_features, params):
        """
        TO DOC
        """
        return cauchy.rvs(params, size=n_features)

    def dirichlet(input_dim, n_features, params):
        """
        TO DOC
        """
        return np.random.dirichlet(params[0] * np.ones(input_dim), n_features)

    def laplace(input_dim, n_features, params):
        """
        TO DOC
        """
        return np.random.laplace(params[0], params[1], (n_features, input_dim))

    def multivariate_normal(input_dim, n_features, params):
        """
        TO DOC
        """
        return np.random.multivariate_normal(np.zeros(input_dim),
                                             np.diag(params), n_features)

    def normal(input_dim, n_features, params):
        """
        TO DOC
        """
        return np.random.normal(0, params[0], (n_features, input_dim))

    def uniform(input_dim, n_features, params):
        """
        TO DOC
        """
        return np.random.uniform(params[0], params[1], (n_features, input_dim))

    ###########################################################################
    ##                                                                       ##
    ## PROJECTION FUNCTION dictionary                                        ##
    ##                                                                       ##
    ## New implementation must be added here.                                ##
    ##                                                                       ##
    ###########################################################################
    __projections = {
        'beta': beta,
        'cauchy': cauchy,
        'dirichlet': dirichlet,
        'laplace': laplace,
        'multivariate_normal': multivariate_normal,
        'normal': normal,
        'uniform': uniform
    }

    def __new__(self, fname):
        # to make the str callable we have to use a dictionary with all the
        # implemented projection matrices
        if fname in self.projections:
            return self.__projections[fname]
        else:
            raise NameError(
                """The name of the projection matrix is not correct or not
                implemented. Check the documentation for all the available
                possibilities.""")

    @classproperty
    def projections(self):
        """
        The available projection matrices.
        """
        return list(self.__projections.keys())
