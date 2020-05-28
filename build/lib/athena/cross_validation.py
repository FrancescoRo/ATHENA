"""Cross validation for Active Subspaces and Non-linear Active Subspaces.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from .active import ActiveSubspaces
from .nas import NonlinearActiveSubspaces
from .gaussian_processes import GPR


def cross_validation(inputs=None,
                     outputs=None,
                     gradients=None,
                     estimator=None,
                     folds=5):
    """n-th fold cross validation from sklearn."""

    inputs = np.hstack((inputs, gradients))
    scores = cross_val_score(estimator,
                             inputs,
                             outputs,
                             cv=folds,
                             scoring=scorer)
    return scores.mean(), scores.std()


class Estimator():
    """
    Estimator needed by sklearn cross-validation.
    See sklearn.model_selection.cross_val_score man.

    Attributes
    ----------
        sstype : string
            defines subspace type to compute, 'AS' or 'NAS'
        n_features : int
            feature space dimension
        pr_matrix : ndarray
            nfeatures-by-m projection matrix onto feature space
        sigma_f : float
            hyperparameter of feature map
        kernel : kernel object
            kernel of the Gaussian process regression
        gp_dimension : int
            dimension of the Response surface built with Gaussian process
            regression
        plot : bool
            switch for plotting the Gaussian process regression
    """
    _fm = None

    def __init__(self,
                 hyperparams=None,
                 sstype=None,
                 n_features=0,
                 feature_map=None,
                 weights=None,
                 method=None,
                 kernel=None,
                 gp_dimension=1,
                 plot=False,
                 title=None):

        self.sstype = sstype
        self.n_features = n_features
        self.feature_map = feature_map
        self.weights = weights
        self.method = method
        self.gp_dimension = gp_dimension
        self.kernel = kernel
        self.hyperparams = hyperparams
        self.plot = plot
        self.title = title

    @classmethod
    def def_feature_map(cls, fm):
        _fm = fm

    def set_params(self, **params):
        pass

    def get_params(self, deep=True):

        p_dict = {
            'sstype': self.sstype,
            'n_features': self.n_features,
            'weights': self.weights,
            'feature_map': self.feature_map,
            'title': self.title,
            'method': self.method,
            'plot': self.plot,
            'gp_dimension': self.gp_dimension,
            'hyperparams': self.hyperparams,
            'kernel': self.kernel
        }

        return p_dict

    def fit(self, X, outputs):
        """Uses Gaussian process regression to build the response surface.
           See sklearn.model_selection.cross_val_score man."""
        print(self._fm)
        inputs, gradients = process_inputs_gradients(X)

        if self.sstype == 'NAS':
            ss = NonlinearActiveSubspaces()
            ss.compute(inputs=inputs,
                       outputs=outputs,
                       gradients=gradients,
                       feature_map=self.feature_map,
                       hyperparams=self.hyperparams,
                       weights=None,
                       method=self.method,
                       nboot=None,
                       n_features=self.n_features)

        elif self.sstype == 'AS':
            ss = ActiveSubspaces()
            ss.compute(inputs, outputs, gradients, self.weights, self.method)

        ss.partition(self.gp_dimension)

        y = ss.forward(inputs)[0]

        if self.kernel is not None:
            gp = gpr(y,
                     outputs,
                     kernel=self.kernel(input_dim=self.gp_dimension, ARD=True))
        else:
            gp = gpr(y, outputs)

        gp.optimize()

        self.gp = gp
        self.ss = ss

        if self.plot is True:
            self.gp.plot()

    def predict(self, X):
        """Predict method of cross-validation.
           See sklearn.model_selection.cross_val_score man."""

        inputs, _ = process_inputs_gradients(X)
        x_test, _ = self.ss.forward(inputs)

        y = self.gp.predict(x_test[:, :self.gp_dimension])[0].reshape(-1)
        return y


def scorer(estimator, X, targets):
    """Score function of cross-validation.
       See sklearn.model_selection.cross_val_score man."""

    y = estimator.predict(X)

    #Normalized Root Mean Square Error
    NRMSE = np.sqrt(np.sum((y-targets.reshape(-1))**2))/\
            np.sqrt(np.sum((targets.reshape(-1)-targets.reshape(-1).mean())**2))

    if estimator.plot is True:
        inputs, _ = process_inputs_gradients(X)
        x_test, _ = estimator.ss.forward(inputs)
        plt.scatter(x_test[:, :estimator.gp_dimension], targets, c=targets)
        plt.show()

    return NRMSE


def process_inputs_gradients(X):
    double_m = X.shape[1]
    m = double_m // 2
    inputs = X[:, :m]
    gradients = X[:, m:]
    return inputs, gradients
