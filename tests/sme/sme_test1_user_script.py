import numpy as np


def the_errors_func(p):
    p_num, p_dim = np.shape(p)

    y_msred = np.array([0.0, 0.0])

    errors_mat = np.ndarray((p_num, 2))
    for i, p_vec in enumerate(p):
        y_model = p_vec
        errors_mat[i, :] = y_model - y_msred

    answer = errors_mat
    return answer
