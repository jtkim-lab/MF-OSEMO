# -*- coding: utf-8 -*-
"""

Created on Mon Aug 19 14:57:25 2019

@author: Syrine Belakaria
This code is based on the code from https://github.com/takeno1995/BayesianOptimization

"""

import numpy as np
import pygmo as pg
import itertools

from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import mfmes as MFBO
import mfmodel as MFGP
import os
from test_functions import branin, lowFidelitybranin, Currin, lowFidelityCurrin
import utils


def get_costs(functions_costs):
    costs = []

    for elem_1 in functions_costs:
        list_cost = []
        cur_cost = -np.inf

        for elem_2 in elem_1:
            assert isinstance(elem_2[1], float)
            assert cur_cost < elem_2[1]

            cur_cost = elem_2[1]
            list_cost.append(elem_2[1])
        costs.append(list_cost)

    return costs

def compute_total_cost(costs, counts_fidelity):
    total_cost = 0.0

    for elem_1, elem_2 in zip(costs, counts_fidelity):
        for cost, count in zip(elem_1, elem_2):
            total_cost += cost / elem_1[-1] * count

    return total_cost

def get_kernels():
    kernel_f = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
    kernel_f.set_params(k1__constant_value_bounds=(1.0, 1.0))

    kernel_e = 0.1 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e2))
    kernel_e.set_params(k1__constant_value_bounds=(0.1, 0.1))

    return kernel_f, kernel_e

def get_mfgp_kernel():
    kernel_f, kernel_e = get_kernels()
    kernel = MFGP.MFGPKernel(kernel_f, kernel_e)

    return kernel


if __name__ == '__main__':
    str_experiment = 'branin_currin_2'
    num_X = 1000
    dim = 2
    num_iter = 10
    num_first = 5
    seed = 0

    functions_costs = [
        [(lowFidelitybranin, 1.0), (branin, 10.0)],
        [(lowFidelityCurrin, 1.0), (Currin, 10.0)],
    ]
    costs = get_costs(functions_costs)

    random_state = np.random.RandomState(seed)

    num_functions = len(functions_costs)
    num_fidelities = [len(elem) for elem in functions_costs]

    #str_approximation = 'TG'
    str_approximation = 'NI'
    assert str_approximation in ['TG', 'NI']

    sample_number = 1
    bound = [0, 1]
    X = random_state.uniform(bound[0], bound[1], size=(num_X, dim))
    Y = []
    counts_fidelity = []

    for elem in functions_costs:
        by = []
        count_fidelity = []

        for ind_elem, value in enumerate(elem):
            func, _ = value

            for bx in X:
                by.append(func(bx, dim))

            if ind_elem == (len(elem) - 1):
                count_fidelity.append(1)
            else:
                count_fidelity.append(num_first)

        Y.append(np.array(by))
        counts_fidelity.append(np.array(count_fidelity))

    total_cost = compute_total_cost(costs, counts_fidelity)
    total_costs_all = [total_cost]

    candidate_x = [np.c_[np.zeros(num_X), X] for i in range(0, num_functions)]

    for ind in range(0, num_functions):
        for m in range(1, num_fidelities[ind]):
            candidate_x[ind] = np.r_[candidate_x[ind], np.c_[m * np.ones(num_X), X]]

    kernel = get_mfgp_kernel()

    ###################GP Initialisation##########################
    GPs = []
    GP_mean = []
    GP_std = []
    cov = []
    MFMES = []
    y_max = []
    GP_index = []
    func_samples = []
    acq_funcs = []

    for i in range(0, num_functions):
        GPs.append(MFGP.MFGPRegressor(kernel=kernel))
        GP_mean.append([])
        GP_std.append([])
        cov.append([])
        MFMES.append(0)
        y_max.append(0)
        temp0 = []

        for m in range(0, num_fidelities[i]):
            temp0 = temp0 + list(random_state.randint(num_X * m, num_X * (m + 1), counts_fidelity[i][m]))

        GP_index.append(np.array(temp0))
    #    GP_index.append(np.random.randint(0, num_X, fir_num))
        func_samples.append([])
        acq_funcs.append([])

    experiment_num=0

    path_file = utils.get_path_file(str_experiment, experiment_num, str_approximation)
    cost_input_output= open(path_file, "a")
    print("total_cost:", total_cost)

    for j in range(0, num_iter):
        if j % 5 != 0:
            for i in range(0, num_functions):
                GPs[i].fit(candidate_x[i][GP_index[i].tolist()], Y[i][GP_index[i].tolist()])
                GP_mean[i], GP_std[i], cov[i] = GPs[i].predict(candidate_x[i])
    #            print("Inference Highest fidelity",GP_mean[i][x_best_index+num_X*(M[i]-1)])

        else:
            for i in range(0, num_functions):
                GPs[i].optimized_fit(candidate_x[i][GP_index[i].tolist()], Y[i][GP_index[i].tolist()])
                GP_mean[i], GP_std[i], cov[i] = GPs[i].optimized_predict(candidate_x[i])

        for i in range(0, num_functions):
            if counts_fidelity[i][num_fidelities[i] - 1] > 0:
                y_max[i] = np.max(Y[i][GP_index[i][GP_index[i] >= (num_fidelities[i] - 1) * num_X]])
            else:
                y_max[i] = GP_mean[i][(num_fidelities[i] - 1) * num_X:][np.argmax(GP_mean[i][(num_fidelities[i] - 1) * num_X:]+GP_std[i][(num_fidelities[i] - 1) * num_X:])]

        # Acquisition function calculation
        for i in range(0, num_functions):
            if str_approximation == 'NI':
                MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch_NI(GP_mean[i], GP_std[i], y_max[i], GP_index[i], num_fidelities[i], costs[i], num_X, cov[i],RegressionModel=GPs[i], sampling_num=sample_number)
            else:
                MFMES[i] = MFBO.MultiFidelityMaxvalueEntroySearch_TG(GP_mean[i], GP_std[i], y_max[i], GP_index[i], num_fidelities[i], costs[i], num_X, RegressionModel=GPs[i], sampling_num=sample_number)

            func_samples[i] = MFMES[i].Sampling_RFM()

        max_samples = []

        for i in range(0, sample_number):
            front = [[-1 * func_samples[k][l][i] for k in range(0, num_functions)] for l in range(0, num_X)] 
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=front)
            cheap_pareto_front = [front[K] for K in ndf[0]]
            maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]

            max_samples.append(maxoffunctions)

        max_samples=list(zip(*max_samples))

        for i in range(0, num_functions):
            acq_funcs[i] = MFMES[i].calc_acq(np.array(max_samples[i]))

        #result[0]values of acq and remaining are the fidelity of each function 
        result = np.zeros((num_X, num_functions + 1))

        for k in range(0, num_X):
            temp=[]

            for i in range(0, num_functions):
                temp.append([acq_funcs[i][k + m * num_X] for m in range(num_fidelities[i])])

            indices = list(itertools.product(*[range(len(x)) for x in temp]))
            values_costs = [sum([float(costs[i][m])/costs[i][num_fidelities[i]-1] for i,m in zip(range(num_functions), index)]) for index in indices]
            values = [float(sum(AF))/i for AF, i in zip(list(itertools.product(*temp)),values_costs)]
            result[k][0] = max(values)
            max_index = np.argmax(values)

            for i in range(0, num_functions):
                result[k][i + 1] = indices[max_index][i]

        x_best_index = np.argmax(list(zip(*result))[0])

        for i in range(0, num_functions):    
            new_index = int(x_best_index + num_X * result[x_best_index][i + 1])
            print(f'new_input {candidate_x[i][new_index]}')

            GP_index[i] = np.r_[GP_index[i], [new_index]]
            counts_fidelity[i][new_index // num_X] += 1

            total_cost = compute_total_cost(costs, counts_fidelity)

        cost_input_output.write(str(total_cost) + ' ' + str(candidate_x[i][new_index]) + ' ' + str(np.array([Y[i][new_index] for i in range(0, num_functions)])) + "\n")
        cost_input_output.close()

        print(f'ITER {j + 1}: total_cost {total_cost:.4f}')
        cost_input_output = open(path_file, "a")
        total_costs_all.append(total_cost)

    cost_input_output.close()
