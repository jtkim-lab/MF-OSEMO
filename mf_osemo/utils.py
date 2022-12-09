import os


def get_path_file(ind_exp, str_approximation, path_outputs='../outputs'):
    str_file = f'{ind_exp}_{str_approximation}_input_output.txt'
    return os.path.join(path_outputs, str_file)
