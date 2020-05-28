"""Tuning procedure for NonlinearActiveSubspaces
"""
from functools import partial
import numpy as np
# from autograd.misc.optimizers import sgd
# from autograd import elementwise_grad as egrad, jacobian
import matplotlib.pyplot as plt
from scipy import optimize
import GPy
import GPyOpt

from .active import ActiveSubspaces
from .kas import NonlinearActiveSubspaces


def tune(ranges,
         par_plot=None,
         opt='brute',
         max_iter=200,
         max_time=3600,
         **kw):
    """doc"""

    # List that collects the parameters evaluted in the optimization
    data = [[], []]
    n_params = len(ranges)

    # function to optimize
    estimator = Estimator(**kw)
    fun_obj = lambda w: Average_RRMSE(w, data=data, estimator=estimator)
    obj = lambda w: obj_autograd(w, kw["inputs"], kw["outputs"], kw["gradients"], kw["n_features"])

    print('#' * 80 + "\nTuning begins")
    if opt == 'brute':
        res, val = optimize.brute(fun_obj, ranges, finish=None,
                                  full_output=True)[:2]
        res = 10**res
    elif opt == 'dual_annealing':
        opt_res = optimize.dual_annealing(fun_obj,
                                          ranges,
                                          maxiter=30,
                                          no_local_search=True)
        res = 10**opt_res.x
        val = opt_res.fun
    elif opt == 'bso':
        bounds = [{
            'name': 'var_' + str(i),
            'type': 'continuous',
            'domain': ranges[i]
        } for i in range(len(ranges))]
        myBopt = GPyOpt.methods.BayesianOptimization(fun_obj,
                                                     domain=bounds,
                                                     model_type='GP',
                                                     acquisition_type='EI',
                                                     exact_feval=True)
        myBopt.run_optimization(max_iter, max_time, eps=1e-16, verbosity=False)
        # myBopt.plot_convergence()
        res = 10**myBopt.x_opt
        val = myBopt.fx_opt
    # elif opt == 'grad':
    #     M, m = kw['inputs'].shape
    #     D = kw['n_features']
    #     res = optimize.minimize(obj,
    #                             np.random.multivariate_normal(
    #                                 np.zeros(m), np.eye(m), D),
    #                             jac=egrad(obj),
    #                             bounds=ranges)
    #     val = res.fun
    #     res = res.x
    # elif opt == 'sgd':
    #     M, m = kw['inputs'].shape
    #     D = kw['n_features']
    #     x = np.random.multivariate_normal(np.zeros(m), np.eye(m), D)
    #     res = sgd(egrad(obj), x)
    #     val = 0
    elif opt == "de":
        result = optimize.differential_evolution(fun_obj, ranges, maxiter=max_iter)
        res = result.x
        val = result.fun

    # plot to show dependance of error from the parameters
    if par_plot:
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
        plt.ylabel("error")

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


def object_fun(params, estimator):
    M, m = estimator.inputs.shape
    D = estimator.n_features

    #compute the projection matrix
    estimator.feature_map.matrix = params.reshape(D, m)

    #compute the score with cross validation for the sampled projection matrix
    score = estimator.training()

    #save the best parameters
    estimator.feature_map.set_best(score)
    print("params {1} mean {0}".format(score, params))
    return score


# def obj_autograd(w, x_, y_, dy_, D):
#     M, m = x_.shape
#     d = y_.shape[1]
#     W = w.reshape(D, m)
#     b = np.random.uniform(0, 2 * np.pi, D)
#     n_tr = np.int(np.floor(M*4/5))
#     x, y, dy = x_[:n_tr, :], y_[:n_tr, :], dy_[:n_tr, :, :]
#     x_t, y_t, dy_t = x_[n_tr:, :], y_[n_tr:, :], dy_[n_tr:, :, :]

#     j_map = jacobian(RFF_map, argnum=1)
#     jac = np.array([j_map(W, x[i, :], b, D, y.mean()) for i in range(n_tr)])
#     ps_jac = np.array([np.linalg.pinv(jac[i, :, :]) for i in range(n_tr)])
#     ps_grad = np.array([np.dot(dy[i, 0, :], ps_jac[i,:, :]) for i in range(n_tr)])
#     cov_matrix = np.array(np.dot(ps_grad.T, ps_grad))
#     #cov_matrix = cov_matrix + 1e-8*np.eye(D)
#     evecs = np.linalg.eigh(cov_matrix)[1]
#     evect = evecs[:, -1]
#     features = np.array([RFF_map(W, x[i, :], b, D, y.mean()) for i in range(n_tr)])
#     features_t = np.array([RFF_map(W, x_t[i, :], b, D, y.mean()) for i in range(x_t.shape[0])])

#     kernel = GPy.kern.RBF(input_dim=1, ARD=True)
#     gp = GPy.models.GPRegression(np.dot(features, evect).reshape(-1, 1), y, kernel)
#     gp.optimize()
#     predictions = gp.predict(np.dot(features_t, evect).reshape(-1, 1))[0]
    # print(gp.param_array)

    #predictions = np.outer(np.dot(features, evect), evect)
    # print(k(x, x_t).shape, np.linalg.inv(k(x, x)).shape)
    #predictions = np.dot(k(x, x_t), np.dot(np.linalg.inv(k(x, x)), y))
    #print(np.dot(features, evect).shape, exact(predictions).shape, predictions.shape)
    #features_t = np.array([RFF_map(W, x_t[i, :], b, D, y.mean()) for i in range(x_t.shape[0])])
    # plt.scatter(np.dot(features, evect), y)
    # plt.scatter(np.dot(features_t, evect), predictions)
    # plt.show()
#     score = np.sqrt(np.sum((y_t - predictions)**2) / np.sum((y_t - y_t.mean())**2))
#     print(score)
#     return score

# def k(x, y):
#     l=0.33072858
#     eps = 0.00886973
#     return 2.31094625*np.exp(-np.array([[np.sum((x[i, :]-y[j, :])**2) for i in range(x.shape[0])] for j in range(y.shape[0])])/l**2)

# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) >= 0)

# def exact(x):
#     return np.array([np.dot(x[i, :].T, x[i, :]) for i in range(x.shape[0])])

# def mapp(W, x):
#     return np.dot(W, x)

# def RFF_map(W, x, b, D, sigma):
#     """doc"""
#     return np.sqrt(2 / D) * sigma * np.cos(np.dot(W, x) + b)

def Average_RRMSE(hyperparams, data, estimator):
    """inputs, outputs, gradients, n_features,
       feature_map, weights, method, kernel, gp_dimension, folds"""

    if len(hyperparams.shape) > 1:
        hyperparams = np.squeeze(hyperparams)
        if len(hyperparams.shape) == 0:
            hyperparams = np.array([hyperparams])

    hyperparams = 10**hyperparams
    tmp = []
    print('#' * 80)
    for it in range(1, 5):
        #compute the projection matrix
        estimator.feature_map.compute(hyperparams)

        #compute the score with cross validation for the sampled projection matrix
        mean, std = estimator.cross_validation()
        #mean, std = estimator.training()

        #save the best parameters
        estimator.feature_map.set_best(mean)
        print("params {2} mean {0}, std {1}".format(mean, std, hyperparams))
        tmp.append(mean)
        if mean > 0.8:
            break
    
    data[0].append(hyperparams)
    data[1].append(estimator.feature_map.score)

    return min(tmp)


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
                 model=None,
                 metric=None,
                 input_cov=None):

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
        self.metric = metric
        self.input_cov = input_cov

    def cross_validation(self):
        """doc"""
        mask = np.arange(self.inputs.shape[0])
        np.random.seed(42)
        np.random.shuffle(mask)
        scores = np.zeros((self.folds))

        # stacked = np.stack(
        #     (self.inputs, self.gradients, self.outputs), axis=0)
        # np.random.shuffle(stacked)
        # scores = np.zeros((self.folds))

        for i in range(self.folds):
            s_mask = np.array_split(mask, self.folds)
            v_mask = s_mask[i]
            validation = (self.inputs[v_mask, :], self.gradients[v_mask, :, :],
                          self.outputs[v_mask, :])
            del s_mask[i]
            t_mask = np.concatenate(s_mask)
            training = (self.inputs[t_mask, :], self.gradients[t_mask, :, :],
                        self.outputs[t_mask, :])

            # splitted = np.array_split(stacked, self.folds)
            # validation = splitted[i]
            # del splitted[i]
            # training = np.vstack(splitted)

            self.fit(*training)
            scores[i] = self.scorer(validation[0], validation[2])

            # self.fit(*self._process_inputs_gradients_outputs(training))
            # scores[i] = self.scorer(validation)

        return scores.mean(), scores.std()

    def training(self):
        """doc"""
        self.fit(self.inputs, self.gradients, self.outputs)
        score = self.scorer(self.inputs, self.outputs)
        return score

    def fit(self, inputs, gradients, outputs):
        """Uses Gaussian process regression to build the response surface.
           See sklearn.model_selection.cross_val_score man."""

        if self.sstype == 'KAS':
            ss = NonlinearActiveSubspaces()
            ss.compute(inputs=inputs,
                       outputs=outputs,
                       gradients=gradients,
                       feature_map=self.feature_map,
                       weights=None,
                       method=self.method,
                       nboot=None,
                       n_features=self.n_features,
                       metric=self.metric,
                       input_cov=self.input_cov)

        elif self.sstype == 'AS':
            ss = ActiveSubspaces()
            ss.compute(inputs=inputs,
                       outputs=outputs,
                       gradients=gradients,
                       weights=self.weights,
                       method=self.method,
                       metric=self.metric,
                       input_cov=self.input_cov)

        # ss.partition(5)
        # np.save("data/modes_"+self.sstype, ss.W1)
        # print("DONE")
        ss.partition(self.gp_dimension)
        y = ss.forward(inputs)[0]
        

        if self.kernel is not None:
            gp = self.gpr(y,
                          outputs,
                          kernel=self.kernel(input_dim=self.gp_dimension,
                                             ARD=True))
        else:
            gp = self.gpr(y, outputs)

        if self.sstype == 'AS':
            np.save("data/training_inputs_AS.npy", y)
            np.save("data/training_outputs_AS.npy", outputs)
            np.save("data/evals_AS.npy", ss.evals)
        else:
            np.save("data/training_inputs_KAS.npy", y)
            np.save("data/training_outputs_KAS.npy", outputs)
            np.save("data/evals_KAS.npy", ss.evals[:10])

        gp.optimize()
        self.gp = gp
        self.ss = ss

        if self.plot is True:
            self.gp.plot()

    def predict(self, inputs):
        """Predict method of cross-validation.
            See sklearn.model_selection.cross_val_score man."""

        # inputs = self._process_inputs_gradients_outputs(validation)[0]
        x_test = self.ss.forward(inputs)[0]

        y = self.gp.predict(x_test[:, :self.gp_dimension])[0]
        return y

    def scorer(self, inputs, targets):
        """Score function of cross-validation.
        See sklearn.model_selection.cross_val_score man."""

        y = self.predict(inputs)
        # targets = self._process_inputs_gradients_outputs(validation)[2]
        
        if targets.shape[1]>2:
            data = np.zeros((targets.shape[1]))
            for i in range(targets.shape[1]):
                data[i] = np.sqrt(np.sum((y[:, i] - targets[:, i])**2) / np.sum((targets[:, i] - targets[:, i].mean())**2))
            error = data.mean()
        else:
            #Normalized Root Mean Square Error
            error = np.sqrt(
                np.sum((y - targets)**2) / np.sum((targets - targets.mean())**2))

        #error = np.sqrt(np.sum((y - targets)**2))
        #error = r2_score(targets˚)

        if self.sstype == 'AS':
            x_test = self.ss.forward(inputs)[0]
            np.save("data/test_inputs_AS.npy", x_test[:, :self.gp_dimension])
            np.save("data/test_outputs_AS.npy", targets[:, 0])
        else:
            x_test = self.ss.forward(inputs)[0]
            np.save("data/test_inputs_KAS.npy", x_test[:, :self.gp_dimension])
            np.save("data/test_outputs_KAS.npy", targets[:, 0])

        if self.plot is True:
            # inputs = self._process_inputs_gradients_outputs(validation)[0]
            x_test = self.ss.forward(inputs)[0]

            if self.gp_dimension == 1:
                for i in range(targets.shape[1]):
                    plt.scatter(x_test[:, :self.gp_dimension],
                                targets[:, i],
                                c=targets[:, i])
            plt.grid()

            if self.sstype == 'KAS':
                plt.xlabel('Nonlinear Active variable ' +
                           r'$W_1^T \phi(\mathbf{x})$',
                           fontsize=14)
                plt.ylabel(r'$f \, (\phi(\mathbf{x}))$', fontsize=14)
            else:
                plt.xlabel('Active variable ' + r'$W_1^T \mathbf{\mu}}$',
                           fontsize=14)
                plt.ylabel(r'$f \, (\mathbf{\mu})$', fontsize=14)

            plt.show()

        return error

    # @staticmethod
    # def _process_inputs_gradients_outputs(X):
    #     double_m_plus_one = X.shape[1]
    #     m = (double_m_plus_one - 1) // 2
    #     inputs = X[:, :m]
    #     gradients = X[:, m:2 * m]
    #     outputs = X[:, -1].reshape(-1, 1)
    #     return inputs, gradients, outputs

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

        gp = GPy.models.GPRegression(x, f, kernel)
        gp.optimize()

        if plot:
            gp.plot()
            plt.show()

        return gp

    # def grad_tuning(self, inputs, outputs, gradients, max_iter, eps):
    #     for i in range(max_iter):
    #         #evaluate score
    #         score, y, gp, ss = self.forward(inputs, outputs, gradients)

    #         if score > eps:
    #             #evaluate gradients
    #             self.backward()
    #         else:
    #             break

    #     return score

    # def forward(self, inputs, outputs, gradients, alpha=1):
    #     ss = NonlinearActiveSubspaces()
    #     ss.compute(inputs=inputs,
    #                 outputs=outputs,
    #                 gradients=gradients,
    #                 feature_map=self.feature_map,
    #                 weights=None,
    #                 method=self.method,
    #                 nboot=None,
    #                 n_features=self.n_features,
    #                 metric=self.metric,
    #                 input_cov=self.input_cov)
    #     ss.partition(self.gp_dimension)
    #     y = ss.forward(inputs)[0]
    #     gp = self.gpr(y, outputs)
    #     gp.optimize()
    #     x_test = ss.forward(inputs)[0]
    #     y = self.gp.predict(x_test[:, :self.gp_dimension])[0]
    #     score = np.sqrt(np.sum((y - outputs)**2) / np.sum((outputs - outputs.mean())**2))
    #     return score, y, gp, ss

    # def backward(self, outputs, predictions, alpha):
    #     batch_size = m
    #     mask = np.random.randint(D*m, size=batch_size)
    #     score_grad = self.compute_grad(mask)#Mxbatch_size
    #     W = self.feature_map.matrix[mask] + alpha * grad_matrix[mask]
    #     self.feature_map.matrix = W

    # def compute_grad(self, mask):
    #     egrad(self.feature_map.jacobian, argnum=2)
    #     return grad(k).dot(K_inv).dot(self.outputs)-k.dot().dot(K_inv).dot(grad(K)).dot(K_inv).dot(self.outputs)

    # def Hadamard(M, W, **kw):  # (nfeatures, (nfeatures, m))
    # """doc"""
    # return M.reshape(kw['n_features'], 1) * W.reshape(kw['n_features'],
    #                                                   kw['input_dim'])

    # def RFF_map(x, W, params, **kw):
    #     """doc"""
    #     return np.sqrt(
    #         2 / kw['n_features']) * kw['sigma_f'] * np.cos(W.dot(x) + params)

    # def RFF_jac(x, W, params, **kw):
    #     """doc"""
    #     return Hadamard(
    #         np.sqrt(2 / kw['n_features']) * kw['sigma_f'] * (-1) *
    #         np.sin(W.dot(x) + params), W, **kw)

    # @staticmethod
    # def Gaussian_kernel(x, variance, lengthscale):
    #     return

    # @staticmethod
    # def feature_map():
    #     pass

    # @staticmethod
    # def eigenfunction():
    #     pass