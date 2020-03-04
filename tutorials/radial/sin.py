#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import GPy
from athena.active import ActiveSubspaces
from athena.nas import  NonlinearActiveSubspaces, tune
from athena.utils import Normalizer
from athena.feature_map import ProjectionMap, RFF_map, RFF_jac
from athena.gaussian_processes import GPR
from athena.cross_validation import Estimator, cross_validation
from radial_functions import *

#Global parameters
M =  800#This is the number of data points to use
m = 2
D = 600
folds = 3

kernel_lap = GPy.kern.RatQuad(input_dim=1, power=1, ARD=True)
kernel_exp = GPy.kern.OU(input_dim=1, ARD=True)

#input ranges
lb = np.array(-3*np.ones(m))
ub = np.array(3*np.ones(m))

#input normalization
XX=np.zeros((M, m))
np.random.seed(42)
for i in range(m):
    XX[:, i] = np.random.uniform(lb[i], ub[i], M)

normalizer = Normalizer(lb, ub)
xx = normalizer.normalize(XX)

#output values (f) and gradients (df)
f = radial(sin, xx, normalizer)
df = radial_grad(dsin, xx, normalizer)

#AS
SS = ActiveSubspaces()
SS.compute(gradients=df, method='exact')
SS.partition(2)
SS.plot_eigenvalues()
SS.plot_sufficient_summary(xx, f)

#AS cross_validation
GPR_AS = Estimator(sstype='AS', weights=None, method='exact', plot=True, gp_dimension=1)
mean, std = cross_validation(inputs=xx, outputs=f, gradients=df, estimator=GPR_AS, folds=3)
print("AS: mean {0}, std {1}".format(mean, std))

#NAS feature map
n_params=1
ranges=[(-1., 1., 0.2) for i in range(n_params)]
b = np.random.uniform(0, 2*np.pi, D)
fm = ProjectionMap(RFF_map, RFF_jac, distr=np.random.multivariate_normal, n_params=n_params,
 input_dim=m, n_features=D, sigma_f=f.var(), b=b)

#NAS tuning
NSS = NonlinearActiveSubspaces()
params_opt, val_opt = tune(inputs=xx, outputs=f, gradients=df, n_features=D, feature_map=fm,
 weights=None, method='exact', ranges=ranges, folds=folds, plot=True, gp_dimension=1, kernel=None)
print("Best params are {0}, corresponding NRMSE is {1}".format(params_opt, val_opt))

#NAS cross_validation
GPR_NAS = Estimator([params_opt],'NAS',D,
                        fm,
                        None,
                        'exact',
                        kernel=None,
                        gp_dimension=1,
                        plot=True)

mean, std = cross_validation(xx, f, df, GPR_NAS, folds=folds)
print("Gaussian: mean {0}, std {1}".format(mean, std))

