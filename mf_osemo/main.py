# -*- coding: utf-8 -*-
"""

Created on Mon Aug 19 14:57:25 2019

@author: Syrine Belakaria
This code is based on the code from https://github.com/takeno1995/BayesianOptimization

"""

import numpy as np
import pygmo as pg
from pygmo import hypervolume
import itertools

from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import mfmes as MFBO
import mfmodel as MFGP
import os
from test_functions import mfbranin, mfCurrin
import utils


functions = [mfbranin, mfCurrin]
str_experiment = 'branin_currin_2'
cost = np.array([[1, 10], [1, 10]])
M = [len(i) for i in cost]

num_X = 1000
dim = 2
num_functions = len(functions)
num_fidelities = cost.shape[1]

assert cost.shape[0] == num_functions

#str_approximation = 'TG'
str_approximation = 'NI'
assert str_approximation in ['TG', 'NI']

fir_num = 5

seed=0
np.random.seed(seed)

sample_number = 1
referencePoint = [1e5] * num_functions
bound = [0, 1]
X = np.random.uniform(bound[0], bound[1], size=(num_X, dim))

# Create data from functions
y = [[] for i in range(0, num_functions)]

for i in range(0, num_functions):
    for m in range(0, num_fidelities):
        for xi in X:
            y[i].append(functions[i](xi, dim, m))

y = [np.asarray(y[i]) for i in range(len(y))]

# Initial setting
fidelity_iter = [np.array([fir_num for j in range(M[i] - 1)] + [1]) for i in range(len(M))]
#fidelity_iter = [np.array([fir_num, fir_num, 1]),np.array([fir_num, 0])]

total_cost = sum([sum([(float(cost[i][m]) / cost[i][M[i] - 1]) * fidelity_iter[i][m] for m in range(M[i])]) for i in range(len(M))])

allcosts = [total_cost]
candidate_x = [np.c_[np.zeros(num_X), X] for i in range(0, num_functions)]

for i in range(0, num_functions):
    for m in range(1, M[i]):
        candidate_x[i] = np.r_[candidate_x[i], np.c_[m*np.ones(num_X), X]]

# Kernel configuration
kernel_f = 1. * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
kernel_f.set_params(k1__constant_value_bounds=(1., 1.))

kernel_e = 0.1 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
kernel_e.set_params(k1__constant_value_bounds=(0.1, 0.1))

kernel = MFGP.MFGPKernel(kernel_f, kernel_e)

###################GP Initialisation##########################
GPs=[]
GP_mean=[]
GP_std=[]
cov=[]
MFMES=[]
y_max=[]
GP_index=[]
func_samples=[]
acq_funcs=[]

for i in range(0, num_functions):
    GPs.append(MFGP.MFGPRegressor(kernel=kernel))
    GP_mean.append([])
    GP_std.append([])
    cov.append([])
    MFMES.append(0)
    y_max.append(0)
    temp0=[]

    for m in range(M[i]):
        temp0=temp0+list(np.random.randint(num_X * m, num_X * (m + 1), fidelity_iter[i][m]))

    GP_index.append(np.array(temp0))
#    GP_index.append(np.random.randint(0, num_X, fir_num))
    func_samples.append([])
    acq_funcs.append([])

experiment_num=0

path_file = utils.get_path_file(str_experiment, experiment_num, str_approximation)
cost_input_output= open(path_file, "a")
print("total_cost:", total_cost)

for j in range(0, 100):
    if j % 5 != 0:
        for i in range(0, num_functions):
            GPs[i].fit(candidate_x[i][GP_index[i].tolist()], y[i][GP_index[i].tolist()])
            GP_mean[i], GP_std[i], cov[i] = GPs[i].predict(candidate_x[i])
#            print("Inference Highest fidelity",GP_mean[i][x_best_index+num_X*(M[i]-1)])

    else:
        for i in range(0, num_functions):
            GPs[i].optimized_fit(candidate_x[i][GP_index[i].tolist()], y[i][GP_index[i].tolist()])
            GP_mean[i], GP_std[i], cov[i] = GPs[i].optimized_predict(candidate_x[i])

    for i in range(0, num_functions):
        if fidelity_iter[i][M[i]-1] > 0:
            y_max[i] = np.max(y[i][GP_index[i][GP_index[i] >= (M[i] - 1) * num_X]])
        else:
            y_max[i] = GP_mean[i][(M[i] - 1) * num_X:][np.argmax(GP_mean[i][(M[i] - 1) * num_X:]+GP_std[i][(M[i] - 1) * num_X:])]

    # Acquisition function calculation
    for i in range(0, num_functions):
        if str_approximation == 'NI':
            MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch_NI(GP_mean[i], GP_std[i], y_max[i], GP_index[i], M[i], cost[i], num_X, cov[i],RegressionModel=GPs[i], sampling_num=sample_number)
        else:
            MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch_TG(GP_mean[i], GP_std[i], y_max[i], GP_index[i], M[i], cost[i], num_X, RegressionModel=GPs[i], sampling_num=sample_number)

        func_samples[i] = MFMES[i].Sampling_RFM()
    max_samples=[]

    for i in range(0, sample_number):
        front=[[-1*func_samples[k][l][i] for k in range(len(functions))] for l in range(num_X)] 
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = front)
        cheap_pareto_front=[front[K] for K in ndf[0]]
        maxoffunctions=[-1*min(f) for f in list(zip(*cheap_pareto_front))]
        max_samples.append(maxoffunctions)
    max_samples=list(zip(*max_samples))

    for i in range(len(functions)):
        acq_funcs[i]=MFMES[i].calc_acq(np.asarray(max_samples[i]))

    #result[0]values of acq and remaining are the fidelity of each function 
    result=np.zeros((num_X, len(functions)+1))

    for k in range(num_X):
        temp=[]
        for i in range(len(functions)):
            temp.append([acq_funcs[i][k + m * num_X] for m in range(M[i])])

        indices=list(itertools.product(*[range(len(x)) for x in temp]))
        values_costs=[sum([float(cost[i][m])/cost[i][M[i]-1] for i,m in zip(range(len(functions)),index)]) for index in indices]
        values=[float(sum(AF))/i for AF,i in zip(list(itertools.product(*temp)),values_costs)]
        result[k][0] = max(values)
        max_index = np.argmax(values)

        for i in range(0, num_functions):
            result[k][i + 1] = indices[max_index][i]

    x_best_index = np.argmax(list(zip(*result))[0])

    for i in range(0, num_functions):    
        new_index = int(x_best_index + num_X * result[x_best_index][i + 1])
        print("new_input",candidate_x[i][new_index])                
        GP_index[i] = np.r_[GP_index[i], [new_index]]
        total_cost += float(cost[i][new_index // num_X])/cost[i][M[i]-1]
        fidelity_iter[i][new_index // num_X] += 1

    cost_input_output.write(str(total_cost)+' '+str(candidate_x[i][new_index])+' '+str(np.array([y[i][new_index] for i in range(len(functions))]))+"\n")
    cost_input_output.close()

    print("total_cost:", total_cost)
    cost_input_output = open(path_file, "a")
    allcosts.append(total_cost)

cost_input_output.close()
