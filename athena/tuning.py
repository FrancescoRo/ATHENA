"""Cross validation for Active Subspaces and Non-linear Active Subspaces.
"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import GPy
from sklearn.model_selection import cross_val_score
from .active import ActiveSubspaces
from .nas import NonlinearActiveSubspaces
from .tools import clock

DEBUG = False


def tune(ranges, plot=None, opt='brute', **kw):
    """Optimize the parameters of the given distribution,
       with respect to the RRMSE. The parameters are optimized with
       logarithmic grid-search.
    """

    # List that collects the parameters evaluted in the optimization in the
    # first component and the corresponding RRMSE in the second.
    data = [[], []]
    n_params = len(ranges)

    # function to optimize
    #fun = partial(Average_RRMSE, data=data, **kw)
    fun = lambda params: Average_RRMSE(params, data, **kw)

    # TODO: provisional design
    if opt == 'brute':
        res, val = optimize.brute(fun, ranges, finish=None, full_output=True)[:2]
        res = 10**res
    elif opt == 'dual_annealing':
        opt_res = optimize.dual_annealing(fun,
                                          ranges,
                                          maxiter=30,
                                          no_local_search=True)
        res = 10**opt_res.x
        val = opt_res.val
    # plot to show dependance of RRMSE from the parameters
    if plot:
        x = np.log10(np.array(data[0][1:]))
        y = np.array(data[1][1:])

        if n_params == 1:
            plt.plot(x, y)
            plt.grid(True, linestyle='dotted')
            plt.xlabel("parameter")
            plt.ylabel("RRMSE")

        elif n_params == 2:
            fig = plt.figure()
            plt.tricontourf(x[:, 0], x[:, 1], y, 15)
            plt.colorbar()
            plt.xlabel("first_hyperparameter")
            plt.ylabel("second_hyperparameter")

        elif n_params == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111,
                                 projection='3d',
                                 title="Best {}".format(res))
            ax.scatter(x[:, 1], x[:, 2], y)
        plt.show()

    kw['feature_map'].set_tuned()

    return res, val


@clock(activate=DEBUG, fmt='[{elapsed:0.8f}s] {name}\n')
def Average_RRMSE(hyperparams, data, inputs, outputs, gradients, n_features,
                  feature_map, weights, method, kernel, gp_dimension, folds):
    """Function to optimize."""
    if hyperparams:
        hyperparams = 10**hyperparams

    for it in range(1, 5):
        print("Iteration with the same sampled #{}".format(it))
        #compute the projection matrix
        feature_map.compute(hyperparams)

        #set the estimator
        GPR_NAS = Estimator('NAS', n_features, feature_map, weights, method,
                            kernel, gp_dimension)

        #compute the score with cross validation for the sampled projection matrix
        mean, std = cross_validation(inputs, outputs, gradients, GPR_NAS,
                                     folds)

        feature_map.set_best(mean)
        print("params {2} mean {0}, std {1}".format(mean, std, hyperparams))

    data[0].append(hyperparams)
    data[1].append(feature_map.score)

    return mean


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
    def __init__(self,
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
        self.kernel = kernel
        self.gp_dimension = gp_dimension
        self.plot = plot
        self.title = title

    def set_params(self, **params):
        pass

    def get_params(self, deep=True):

        dic = {
            'sstype': self.sstype,
            'n_features': self.n_features,
            'feature_map': self.feature_map,
            'weights': self.weights,
            'method': self.method,
            'kernel': self.kernel,
            'gp_dimension': self.gp_dimension,
            'plot': self.plot,
            'title': self.title,
        }

        return dic

    def fit(self, X, outputs):
        """Uses Gaussian process regression to build the response surface.
           See sklearn.model_selection.cross_val_score man."""

        inputs, gradients = process_inputs_gradients(X)

        if self.sstype == 'NAS':
            ss = NonlinearActiveSubspaces()
            ss.compute(inputs=inputs,
                       outputs=outputs,
                       gradients=gradients,
                       feature_map=self.feature_map,
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
            gp = GPR(y,
                     outputs,
                     kernel=self.kernel(input_dim=self.gp_dimension, ARD=True))
        else:
            gp = GPR(y, outputs)

        gp.optimize()

        self.gp = gp
        self.ss = ss

        if self.plot is True:
            self.gp.plot()

    def predict(self, X):
        """Predict method of cross-validation.
           See sklearn.model_selection.cross_val_score man."""

        inputs = process_inputs_gradients(X)[0]
        x_test = self.ss.forward(inputs)[0]

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

def GPR(x, f, kernel=None, plot=False):
    """Execute Gaussian processes regression for input data x and targets f.
    Returns
    -------
    gp : gaussian process instance
        gaussian process instance from the library GPy
    """
    m = x.shape[1]
    
    if kernel is None:
        kernel = GPy.kern.RBF(input_dim=m, ARD=True)

    gp = GPy.models.GPRegression(x, f.reshape(-1, 1), kernel)
    gp.optimize()

    if plot:
        gp.plot()
        plt.show()

    return gp