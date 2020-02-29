"""[summary]

[description]
"""
import numpy as np

class Feature_map():
    """Active Subspaces base class

    [description]
    """

	def __init__(fm):
		fm.map = None
        fm.jacobian = None
        fm.n_features = None
        fm.input_dim = None
        fm.hyperparams = None

    def compute(fm,
                map=None,
                jacobian=None,
                n_features=None,
                input_dim=None,
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

	def map(fm, x, **hyperparams):
		return fm.map(x, *hyperparams)

	def feature_map_Jac(fm, x, *params):
	    params=params[0]
	    return fm.jacobian(x, *hyperparams)
