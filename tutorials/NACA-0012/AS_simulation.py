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
from collections import namedtuple
from functools import partial


M = 300
D = 2000
m = 7
folds = 3
dim = 1

samples=np.loadtxt("reference_samples_Ulisse")
samples2=np.loadtxt("reference_samples_Ulisse_2")
samples3=np.loadtxt("reference_samples_Ulisse_3")
x=np.vstack((samples, samples2, samples3))

#output values (f) and gradients (df)
output = np.loadtxt("output")
output2 = np.loadtxt("output2")
output3 = np.loadtxt("output3")
f = np.vstack((output, output2, output3))[:, 0].reshape(-1)#Cd
#Cl = np.vstack((output, output2, output3))[:, 1].reshape(-1, 1)

#compute gradients with gaussian processes
gp = GPR(x, f.reshape(-1, 1))
gp.optimize()
df = gp.predict_jacobian(x)[0].reshape(M, m)

#AS
ss = ActiveSubspaces()
ss.compute(gradients=df)
ss.partition(2)
ss.plot_eigenvalues()
ss.plot_sufficient_summary(x, f)
labels = ['U', 'ni', 'x0', 'y0', 'alpha', 'upper', 'lower']

#AS cross validation
GPR_AS = Estimator(sstype='AS', weights=None, method='exact', plot=True, gp_dimension=dim)
mean_AS, std_AS = cross_validation(inputs=x, outputs=f, gradients=df, estimator=GPR_AS,
 folds=folds)
file = open("Results.txt", 'a')
file.write("AS mean {0}, std {1}\n".format(mean_AS, std_AS))
file.close()

def tune_distr(namedDistr):

	n_params= namedDistr.n_params
	distr = namedDistr.distr
	ranges = namedDistr.ranges
	kernel = namedDistr.kernel
	name = namedDistr.name

	#NAS feature map
	b = np.random.uniform(0, 2*np.pi, D)
	fm = ProjectionMap(RFF_map, RFF_jac, distr=distr,
	 n_params=n_params, input_dim=m, n_features=D, sigma_f=f.var(), b=b)

	#NAS tune
	NSS = NonlinearActiveSubspaces()
	params_opt, val_opt = tune(inputs=x, outputs=f, gradients=df, n_features=D, feature_map=fm,
	 weights=None, method='exact', ranges=ranges, folds=folds, plot=False, gp_dimension=1,
	 kernel=kernel, opt='brute')

	file = open("Results.txt", 'a')
	file.write("Best params for {0} are\n{1} corresponding NRMSE is {2}\n".format(name, params_opt, val_opt))
	file.close()

Distr = namedtuple('Distr', 'name n_params distr ranges kernel')
range_1 = [(-3., 1., 0.2)]
range_2 = [(-3., 1., 0.5) for i in range(2)]

#kernels for GPR
RBF = GPy.kern.RBF
Lap_k = GPy.kern.RatQuad
Cauchy = GPy.kern.OU
Exp = GPy.kern.Exponential
Mat32 = GPy.kern.Matern32
Mat52 = GPy.kern.Matern52

#distr
normal = np.random.multivariate_normal
laplace = np.random.laplace
beta = np.random.beta

#Named distr
Gaussian = Distr('Gaussian_RBF', 1, normal, range_1, RBF)
Gaussian_Laplace = Distr('Gaussian_Laplace',1, normal,  range_1, Lap_k)
Gaussian_Cauchy = Distr('Gaussian_Cauchy',1, normal,  range_1, Cauchy)
Gaussian_Exp = Distr('Gaussian_Exp',1, normal,  range_1, Exp)
Gaussian_Mat32 = Distr('Gaussian_Mat32',1, normal,  range_1, Mat32)
Gaussian_Mat52 = Distr('Gaussian_Mat52',1, normal,  range_1, Mat52)
Laplace = Distr('Laplace_RBF',1, laplace, range_1, RBF)
Laplace_Laplace = Distr('Laplace_Laplace',1, laplace, range_1, Lap_k)
Beta = Distr('Beta_RBF',2, beta, range_2, RBF)
Beta_Laplace = Distr('Beta_Laplace',2, beta, range_2, Lap_k)

lis = [Gaussian, Gaussian_Laplace, Laplace, Laplace_Laplace, Beta, Beta_Laplace]

for it in lis:
	tune_distr(it)

"""
NRMSE
Lift
AS GPR: mean 0.369388586553, std 0.049972351853
Gaussian Laplace kernel: mean 0.350591700853, std 0.0496024930501
Gaussian: mean 0.346241340481, std 0.0532948717831
Laplace Laplace kernel: mean 0.350264726035, std 0.0477351009997
Laplace:mean 0.352505153603, std 0.0549001879225
Beta Laplace kernel: mean 0.364776817943, std 0.0508608330725
Beta: mean 0.366643855249, std 0.0450949779542
Drag
AS GPR: mean 0.256423606886, std 0.0369129873747
Gaussian Laplace kernel: mean 0.237710750742, std 0.035200825978
Gaussian: mean 0.24526556745, std 0.0334901097299
Laplace Laplace kernel: mean 0.234276087361, std 0.0305681990794
Laplace:mean 0.23318997795, std 0.0307063605946
Beta Laplace kernel: mean 0.226505444841, std 0.023306956606
Beta: mean 0.229606795692, std 0.0250377012839
"""
