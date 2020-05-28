#!/usr/bin/env python
# coding: utf-8

import numpy as np
import GPy
from athena.active import ActiveSubspaces
from athena.nas import NonlinearActiveSubspaces
from athena.utils import Normalizer
from athena.feature_map import FeatureMap, RFF_map, RFF_jac
from athena.tuning import tune, Estimator
from radial_functions import radial, radial_grad, paraboloid, dparaboloid, sin, dsin
import matplotlib.pyplot as plt
# Global parameters
M = 300
m = 3
D = 700
folds = 3

kernel_lap = GPy.kern.RatQuad(input_dim=1, power=1, ARD=True)
kernel_exp = GPy.kern.OU(input_dim=1, ARD=True)

# input ranges
lb = np.array(-1 * np.ones(m))
ub = np.array(1 * np.ones(m))

# input normalization
XX = np.zeros((M, m))
np.random.seed(42)
for i in range(m):
    XX[:, i] = np.random.uniform(lb[i], ub[i], M)

normalizer = Normalizer(lb, ub)
xx = normalizer.normalize(XX)

# output values (f) and gradients (df)
f = radial(paraboloid, xx, normalizer).reshape(-1, 1)
df = radial_grad(dparaboloid, xx, normalizer).reshape(M, 1, m)
# g = radial(sin, xx, normalizer).reshape(-1, 1)
# dg = radial_grad(dsin, xx, normalizer).reshape(M, 1, m)
# f = np.squeeze(np.stack((f, g), axis=1))
# df = np.squeeze(np.stack((df, dg), axis=1))

#AS
ss = ActiveSubspaces()
ss.compute(inputs=xx,
           gradients=df,
           method='exact',
           outputs=f)
ss.partition(2)
ss.plot_eigenvalues()
# ss.plot_sufficient_summary(xx, f[:, 1])
# ss.plot_sufficient_summary(xx, f[:, 0])

ss.partition(1)
y = ss.forward(xx)[0]
print(y.shape, f.shape)
kernel = GPy.kern.RBF(input_dim=1, ARD=True)
gp = GPy.models.GPRegression(X=y, Y=f, kernel=kernel)
gp = GPy.models.GPRegression(X=y, Y=f[:, 1].reshape(-1, 1), kernel=kernel)
gp.optimize()
gp.plot()
plt.show()
gp = GPy.models.GPRegression(X=y, Y=f[:, 0].reshape(-1, 1), kernel=kernel)
gp.optimize()
gp.plot()
plt.show()

# AS cross validation
GPR_AS = Estimator(sstype='AS',
                   weights=None,
                   method='exact',
                   plot=True,
                   gp_dimension=1,
                   inputs=xx,
                   outputs=f,
                   gradients=df,
                   folds=3)

mean, std = GPR_AS.cross_validation()
print("AS: mean {0}, std {1}".format(mean, std))

# NAS feature map
n_params = 1
ranges = [(-3., 1., 0.2) for i in range(n_params)]
b = np.random.uniform(0, 2 * np.pi, D)
fm = FeatureMap(RFF_map,
                RFF_jac,
                distr=np.random.normal,
                n_params=n_params,
                input_dim=m,
                n_features=D,
                sigma_f=f.var(),
                b=b)

# NAS tune
ranges = [(-2, 1)]
params_opt, val_opt = tune(inputs=xx,
                           outputs=f,
                           gradients=df,
                           n_features=D,
                           feature_map=fm,
                           weights=None,
                           method='exact',
                           ranges=ranges,
                           folds=2,
                           plot=True,
                           gp_dimension=1,
                           kernel=None,
                           sstype='NAS')

print("Best params are {0}, corresponding NRMSE is {1}".format(
    params_opt, val_opt))
print("Is feature map tuned? {}".format(fm.tuned))

# NAS cross_validation
GPR_NAS = Estimator(inputs=xx,
                    outputs=f,
                    gradients=df,
                    sstype='NAS',
                    n_features=D,
                    feature_map=fm,
                    weights=None,
                    method='exact',
                    kernel=None,
                    gp_dimension=1,
                    plot=True)

mean, std = GPR_NAS.cross_validation()
print("Gaussian: mean {0}, std {1}".format(mean, std))
