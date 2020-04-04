"""Tuning procedure for NonlinearActiveSubspaces
"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import GPy

from .active import ActiveSubspaces
from .nas import NonlinearActiveSubspaces


def tune(ranges, plot=None, opt='brute', **kw):
    """doc"""

    # List that collects the parameters evaluted in the optimization
    data = [[], []]
    n_params = len(ranges)

    # function to optimize
    estimator = Estimator(**kw)
    fun = partial(Average_RRMSE, data=data, estimator=estimator)

    print('#' * 80 + "\nTuning begins")
    if opt == 'brute':
        res, val = optimize.brute(fun, ranges, finish=None,
                                  full_output=True)[:2]
        res = 10**res
    elif opt == 'dual_annealing':
        opt_res = optimize.dual_annealing(fun,
                                          ranges,
                                          maxiter=30,
                                          no_local_search=True)
        res = 10**opt_res.x
        val = opt_res.fun

    # plot to show dependance of NRMSE from the parameters
    if plot:
        plot_tune_results(data, n_params, res)

    kw['feature_map'].set_tuned()
    print('#' * 80 + "\nTuning is completed\n" + '#' * 80)
    return res, val


def plot_tune_results(data, n_params, res):
    """doc"""
    x = np.log10(np.array(data[0][1:]))
    y = np.array(data[1][1:])

    if n_params == 1:
        plt.plot(x, y)
        plt.grid(True, linestyle='dotted')
        plt.xlabel("parameter")
        plt.ylabel("NRMSE")

    elif n_params == 2:
        fig = plt.figure()
        plt.tricontourf(x[:, 0], x[:, 1], y, 15)
        plt.colorbar()
        plt.xlabel("first_hyperparameter")
        plt.ylabel("second_hyperparameter")

    elif n_params == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', title="Best {}".format(res))
        ax.scatter(x[:, 1], x[:, 2], y)

    plt.show()

def Average_RRMSE(hyperparams, data, estimator):
    """inputs, outputs, gradients, n_features,
       feature_map, weights, method, kernel, gp_dimension, folds"""

    hyperparams = 10**hyperparams

    print('#' * 80)
    for it in range(1, 5):
        #compute the projection matrix
        estimator.feature_map.compute(hyperparams)

        #compute the score with cross validation for the sampled projection matrix
        mean, std = estimator.cross_validation()

        #save the best parameters
        estimator.feature_map.set_best(mean)
        print("params {2} mean {0}, std {1}".format(mean, std, hyperparams))

    data[0].append(hyperparams)
    data[1].append(estimator.feature_map.score)

    return mean

class Estimator():
    """doc"""
    def __init__(self,
                 inputs=None,
                 outputs=None,
                 gradients=None,
                 folds=5,
                 sstype=None,
                 n_features=0,
                 feature_map=None,
                 weights=None,
                 method=None,
                 kernel=None,
                 gp_dimension=1,
                 plot=False,
                 model=None):

        self.inputs = inputs
        self.outputs = outputs
        self.gradients = gradients
        self.folds = folds
        self.sstype = sstype
        self.n_features = n_features
        self.feature_map = feature_map
        self.weights = weights
        self.method = method
        self.kernel = kernel
        self.gp_dimension = gp_dimension
        self.plot = plot
        self.ss = None
        self.gp = None
        self.model = model

    def cross_validation(self):
        """doc"""
        stacked = np.hstack(
            (self.inputs, self.gradients, self.outputs.reshape(-1, 1)))
        np.random.shuffle(stacked)
        scores = np.zeros((self.folds))

        for i in range(self.folds):
            splitted = np.array_split(stacked, self.folds)
            validation = splitted[i]
            del splitted[i]
            training = np.vstack(splitted)

            self.fit(*self._process_inputs_gradients_outputs(training))
            scores[i] = self.scorer(validation)

        return scores.mean(), scores.std()

    def fit(self, inputs, gradients, outputs):
        """Uses Gaussian process regression to build the response surface.
           See sklearn.model_selection.cross_val_score man."""

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
            gp = self.gpr(y,
                          outputs,
                          kernel=self.kernel(input_dim=self.gp_dimension,
                                             ARD=True))
        else:
            gp = self.gpr(y, outputs)

        gp.optimize()

        self.gp = gp
        self.ss = ss

        if self.plot is True:
            self.gp.plot()

    def predict(self, validation):
        """Predict method of cross-validation.
            See sklearn.model_selection.cross_val_score man."""

        inputs = self._process_inputs_gradients_outputs(validation)[0]
        x_test = self.ss.forward(inputs)[0]

        y = self.gp.predict(x_test[:, :self.gp_dimension])[0].reshape(-1)
        return y

    def scorer(self, validation):
        """Score function of cross-validation.
        See sklearn.model_selection.cross_val_score man."""

        y = self.predict(validation)
        targets = self._process_inputs_gradients_outputs(validation)[2]

        #Normalized Root Mean Square Error
        # NRMSE = np.sqrt(np.sum((y-targets.reshape(-1))**2))/\
        #         np.sqrt(np.sum((targets.reshape(-1)-targets.reshape(-1).mean())**2))
        NRMSE = np.sqrt(np.sum((y-targets.reshape(-1))**2))

        if self.plot is True:
            inputs = self._process_inputs_gradients_outputs(validation)[0]
            x_test = self.ss.forward(inputs)[0]
            plt.scatter(x_test[:, :self.gp_dimension], targets, c=targets)
            plt.grid()

            if self.sstype == 'NAS':
                plt.xlabel('Nonlinear Active variable ' + r'$W_1^T \phi(\mathbf{x})$',
                        fontsize=14)
                plt.ylabel(r'$f \, (\phi(\mathbf{x}))$', fontsize=14)
            else:
                plt.xlabel('Active variable ' + r'$W_1^T \mathbf{\mu}}$',
                       fontsize=14)
                plt.ylabel(r'$f \, (\mathbf{\mu})$', fontsize=14)

            plt.show()
            
        return NRMSE

    @staticmethod
    def _process_inputs_gradients_outputs(X):
        double_m_plus_one = X.shape[1]
        m = (double_m_plus_one - 1) // 2
        inputs = X[:, :m]
        gradients = X[:, m:2 * m]
        outputs = X[:, -1].reshape(-1, 1)
        return inputs, gradients, outputs

    @staticmethod
    def gpr(x, f, kernel=None, plot=False):
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