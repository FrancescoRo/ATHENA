"""
Athena init
"""
__all__ = ['active', 'kas', 'subspaces', 'utils', 'feature_map', 'tuning']
__project__ = 'ATHENA'
__title__ = "athena"
__author__ = "Marco Tezzele, Francesco Romor"
__copyright__ = "Copyright 2019-2020, Athena contributors"
__license__ = "MIT"
__version__ = "0.0.1"
__mail__ = 'marcotez@gmail.com, francesco.romor@gmail.com'
__maintainer__ = __author__
__status__ = "Beta"

from .active import ActiveSubspaces
from .kas import NonlinearActiveSubspaces
from .subspaces import Subspaces
from .utils import (Normalizer, initialize_weights, linear_program_ineq,
                    local_linear_gradients, sort_eigpairs)
from .tuning import (tune, Estimator)
from .feature_map import (FeatureMap)
