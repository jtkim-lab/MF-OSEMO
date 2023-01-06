"""

Created on Mon Aug 19 14:57:25 2019

@author: Syrine Belakaria
This code is based on the code from https://github.com/takeno1995/BayesianOptimization

"""

import numpy as np
import time
import pygmo as pg
import itertools

from sklearn.gaussian_process.kernels import RBF

import mf_osemo.mfmes as MFBO
import mf_osemo.mfmodel as MFGP


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

def get_initializations(functions_costs, X, seed, num_init_low=5, num_init_high=1):
    random_state = np.random.RandomState(seed)
    num_X = X.shape[0]

    queries = []
    queries_indices = []
    evaluations = []
    counts_fidelity = []

    for ind_elem, elem in enumerate(functions_costs):
        queries_ = []
        queries_indices_ = []
        evaluations_ = []
        count_fidelity = []

        for ind_value, value in enumerate(elem):
            func, _ = value
            if ind_value == (len(elem) - 1):
                num_init_ = num_init_high
            else:
                num_init_ = num_init_low

            indices_ = random_state.choice(np.arange(0, num_X), size=num_init_, replace=False)

            X_ = X[indices_]

            for bx_ in X_:
                queries_.append([ind_value] + list(bx_))
                evaluations_.append(func(bx_))

            queries_indices_ += list(indices_ + num_X * ind_value)
            count_fidelity.append(num_init_)

        queries.append(np.array(queries_))
        queries_indices.append(np.array(queries_indices_))
        evaluations.append(np.array(evaluations_))
        counts_fidelity.append(np.array(count_fidelity))

    return queries, queries_indices, evaluations, counts_fidelity

def get_X(bounds, num_X, seed):
    random_state = np.random.RandomState(seed)
    dim = bounds.shape[0]

    X = (bounds[:, 1] - bounds[:, 0]) * random_state.uniform(low=0.0, high=1.0, size=(num_X, dim)) + bounds[:, 0]

    return X

def get_candidates(X, num_functions, num_fidelities):
    num_X = X.shape[0]
    candidates = [np.c_[np.zeros(num_X), X] for _ in range(0, num_functions)]

    for ind in range(0, num_functions):
        for m in range(1, num_fidelities[ind]):
            candidates[ind] = np.r_[
                candidates[ind],
                np.c_[m * np.ones(num_X), X]
            ]

    return candidates

def get_max_samples(func_samples, num_X, num_functions, sample_number):
    max_samples = []

    for i in range(0, sample_number):
        front = [[-1 * func_samples[k][l][i] for k in range(0, num_functions)] for l in range(0, num_X)]
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=front)
        cheap_pareto_front = [front[K] for K in ndf[0]]
        maxoffunctions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]

        max_samples.append(maxoffunctions)

    max_samples = list(zip(*max_samples))
    return max_samples

def run_mf_mo_bo_iter(
    ind_iter,
    GPs,
    queries,
    queries_indices,
    evaluations,
    counts_fidelity,
    total_costs,
    num_X,
    candidate_X,
    dict_info,
):
    time_start = time.time()

    functions_costs = dict_info['functions_costs']
    costs = dict_info['costs']
    num_functions = dict_info['num_functions']
    num_fidelities = dict_info['num_fidelities']
    str_approximation = dict_info['str_approximation']
    sample_number = dict_info['sample_number']

    func_samples = []
    MFMES = []

    for ind in range(0, num_functions):
        cost_ = costs[ind]

        if ind_iter % 5 != 0:
            GPs[ind].fit(
                queries[ind],
                evaluations[ind],
            )
            mean_, std_, cov_ = GPs[ind].predict(candidate_X[ind])
        else:
            GPs[ind].optimized_fit(
                queries[ind],
                evaluations[ind],
            )
            mean_, std_, cov_ = GPs[ind].optimized_predict(
                candidate_X[ind])

        if counts_fidelity[ind][num_fidelities[ind] - 1] > 0:
            y_max_ = np.max(evaluations[ind][queries[ind][:, 0] == (num_fidelities[ind] - 1)])
        else:
            y_max_ = mean_[(num_fidelities[ind] - 1) * num_X:][np.argmax(mean_[(num_fidelities[ind] - 1) * num_X:] + std_[(num_fidelities[ind] - 1) * num_X:])]

        if str_approximation == 'NI':
            MFMES.append(MFBO.MultiFidelityMaxvalueEntroySearch_NI(mean_, std_, y_max_, queries_indices[ind], num_fidelities[ind], cost_, num_X, cov_, RegressionModel=GPs[ind], sampling_num=sample_number))
        elif str_approximation == 'TG':
            MFMES.append(MFBO.MultiFidelityMaxvalueEntroySearch_TG(mean_, std_, y_max_, queries_indices[ind], num_fidelities[ind], cost_, num_X, RegressionModel=GPs[ind], sampling_num=sample_number))
        else:
            raise ValueError

    for ind in range(0, num_functions):
        func_samples.append(MFMES[ind].Sampling_RFM())

    max_samples = get_max_samples(
        func_samples, num_X, num_functions, sample_number)

    acq_funcs = []

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

    for ind in range(0, num_functions):
        new_index = int(x_best_index + num_X * result[x_best_index][ind + 1])
        print(f'new_input {candidate_X[ind][new_index]}')

        queries_indices[ind] = np.concatenate([queries_indices[ind], [new_index]], axis=0)
        next_point = candidate_X[ind][new_index]
        next_evaluation = functions_costs[ind][int(next_point[0])][0](next_point[1:])

        queries[ind] = np.concatenate(
            [queries[ind], [next_point]],
            axis=0
        )
        evaluations[ind] = np.concatenate(
            [evaluations[ind], [next_evaluation]],
            axis=0
        )
        counts_fidelity[ind][new_index // num_X] += 1

        total_cost = compute_total_cost(costs, counts_fidelity)

    total_costs.append(total_cost)
    time_end = time.time()
    time_consumed = time_end - time_start

    print(f'ITER {ind_iter + 1:04d}: total_cost {total_cost:.4f} time_consumed {time_consumed:.4f}')

    return GPs, queries, queries_indices, evaluations, counts_fidelity, total_costs, time_consumed

def run_mf_mo_bo(
    functions_costs,
    bounds,
    num_iter,
    num_X,
    str_approximation,
    sample_number,
    seed,
):
    assert isinstance(functions_costs, list)
    assert isinstance(bounds, np.ndarray)
    assert isinstance(num_iter, int)
    assert isinstance(num_X, int)
    assert isinstance(str_approximation, str)
    assert isinstance(sample_number, int)
    assert isinstance(seed, int)
    assert len(bounds.shape) == 2
    assert bounds.shape[1] == 2
    assert str_approximation in ['TG', 'NI']

    costs = get_costs(functions_costs)

    num_functions = len(functions_costs)
    num_fidelities = [len(elem) for elem in functions_costs]

    X = get_X(bounds, num_X, seed)
    candidate_X = get_candidates(X, num_functions, num_fidelities)

    queries, queries_indices, evaluations, counts_fidelity = get_initializations(functions_costs, X, seed)

    total_cost = compute_total_cost(costs, counts_fidelity)
    total_costs = [total_cost]
    print(f'initial total_cost {total_cost:.4f}')

    dict_info = {
        'functions_costs': functions_costs,
        'costs': costs,
        'num_functions': num_functions,
        'num_fidelities': num_fidelities,
        'str_approximation': str_approximation,
        'sample_number': sample_number,
    }

    kernel = get_mfgp_kernel()
    GPs = []

    for ind in range(0, num_functions):
        GPs.append(MFGP.MFGPRegressor(kernel=kernel))

    for ind_iter in range(0, num_iter):
        GPs, queries, queries_indices, evaluations, counts_fidelity, total_costs, time_consumed = run_mf_mo_bo_iter(
            ind_iter,
            GPs,
            queries,
            queries_indices,
            evaluations,
            counts_fidelity,
            total_costs,
            num_X,
            candidate_X,
            dict_info,
        )
