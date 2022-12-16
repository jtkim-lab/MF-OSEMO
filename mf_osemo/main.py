import numpy as np

import run_bo
from test_functions import branin, lowFidelitybranin, Currin, lowFidelityCurrin


if __name__ == '__main__':
    bounds = np.array([
        [0, 1],
        [0, 1],
    ])

    functions_costs = [
        [(lowFidelitybranin, 1.0), (branin, 10.0)],
        [(lowFidelityCurrin, 1.0), (Currin, 10.0)],
    ]

    num_iter = 50
    num_X = 1000
#    str_approximation = 'TG'
    str_approximation = 'NI'
    sample_number = 1
    seed = 42

    run_bo.run_mfmo_bo(
        functions_costs,
        bounds,
        num_iter,
        num_X,
        str_approximation,
        sample_number,
        seed,
    )
