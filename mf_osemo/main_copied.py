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
    num_experiment = 0

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

    candidate_X = [np.c_[np.zeros(num_X), X] for i in range(0, num_functions)]

    for ind in range(0, num_functions):
        for m in range(1, num_fidelities[ind]):
            candidate_X[ind] = np.r_[candidate_X[ind], np.c_[m * np.ones(num_X), X]]

    kernel = get_mfgp_kernel()

    GPs = []
    GP_index = []

    for ind in range(0, num_functions):
        GPs.append(MFGP.MFGPRegressor(kernel=kernel))
        temp0 = []

        for m in range(0, num_fidelities[ind]):
            temp0 += list(random_state.randint(
                num_X * m, num_X * (m + 1),
                counts_fidelity[ind][m]
            ))

        GP_index.append(np.array(temp0))

    print("total_cost:", total_cost)

    for ind_iter in range(0, num_iter):
        y_max = []
        func_samples = []
        acq_funcs = []
        max_samples = []
        MFMES = []

        for ind in range(0, num_functions):
            cost_ = costs[ind]

            if ind_iter % 5 != 0:
                GPs[ind].fit(
                    candidate_X[ind][GP_index[ind].tolist()],
                    Y[ind][GP_index[ind].tolist()]
                )
                mean_, std_, cov_ = GPs[ind].predict(candidate_X[ind])
            else:
                GPs[ind].optimized_fit(
                    candidate_X[ind][GP_index[ind].tolist()],
                    Y[ind][GP_index[ind].tolist()]
                )
                mean_, std_, cov_ = GPs[ind].optimized_predict(
                    candidate_X[ind])

            if counts_fidelity[ind][num_fidelities[ind] - 1] > 0:
                y_max.append(np.max(Y[ind][GP_index[ind][GP_index[ind] >= (num_fidelities[ind] - 1) * num_X]]))
            else:
                y_max.append(mean_[(num_fidelities[ind] - 1) * num_X:][np.argmax(mean_[(num_fidelities[ind] - 1) * num_X:] + std_[(num_fidelities[ind] - 1) * num_X:])])

            if str_approximation == 'NI':
                MFMES.append(MFBO.MultiFidelityMaxvalueEntroySearch_NI(mean_, std_, y_max[ind], GP_index[ind], num_fidelities[ind], cost_, num_X, cov_,RegressionModel=GPs[ind], sampling_num=sample_number))
            elif str_approximation == 'TG':
                MFMES.append(MFBO.MultiFidelityMaxvalueEntroySearch_TG(mean_, std_, y_max[ind], GP_index[ind], num_fidelities[ind], cost_, num_X, RegressionModel=GPs[ind], sampling_num=sample_number))
            else:
                raise ValueError

            func_samples.append(MFMES[ind].Sampling_RFM())

        for i in range(0, sample_number):
            front = [[-1 * func_samples[k][l][i] for k in range(0, num_functions)] for l in range(0, num_X)]
            ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=front)
            cheap_pareto_front = [front[K] for K in ndf[0]]
            maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]

            max_samples.append(maxoffunctions)

        max_samples = list(zip(*max_samples))

        for i in range(0, num_functions):
            acq_funcs.append(MFMES[i].calc_acq(np.array(max_samples[i])))

        #result[0]values of acq and remaining are the fidelity of each function 
        result = np.zeros((num_X, num_functions + 1))

        for k in range(0, num_X):
            temp=[]

            for ind in range(0, num_functions):
                temp.append([acq_funcs[ind][k + m * num_X] for m in range(0, num_fidelities[ind])])

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
            print(f'new_input {candidate_X[i][new_index]}')

            GP_index[i] = np.r_[GP_index[i], [new_index]]
            counts_fidelity[i][new_index // num_X] += 1

            total_cost = compute_total_cost(costs, counts_fidelity)

        print(f'ITER {ind_iter + 1}: total_cost {total_cost:.4f}')
        total_costs_all.append(total_cost)
