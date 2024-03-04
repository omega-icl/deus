import numpy as np
import time
'''
Design Space - Test 2:
Find a probabilistic design space of a given reliability value 'a'.
The model is the following:
    s = p1*d1^2 + d2
    , where: 
    d1, d2 are design variables;
    p1 is a model parameter that has an uncertain value described by a
    Gaussian distribution, N(m, sigma).
The constraints that must be fulfilled are the following:
    0.2 <= s <= 0.75  
'''


class Model:
    def __init__(self):
        pass

    def s(self, d, p):
        time.sleep(0.0001)
        d1, d2 = d
        p1 = p[0]
        x1 = p1*d1**2 + d2
        return x1


def g_func(d, p):
    '''
    :param d: 2d array, each row is a design vector
    :param p: 2d array, each row is a parameter vector
    :return: list of 2d array, j^th row in i^th item is g(d_i, p_j)
    '''

    n_p, p_dims = np.shape(p)
    g_dims = 2

    a_model = Model()
    g_list = []
    for i, d_vec in enumerate(d):
        g_mat = np.empty((n_p, g_dims))
        for j, p_vec in enumerate(p):
            s = a_model.s(d_vec, p_vec)
            g1 = s - 0.2
            g2 = 0.75 - s
            g_mat[j, :] = np.array([g1, g2])

        g_list.append(g_mat)
    return g_list


if __name__ == "__main__":
    np.random.seed(1)

    dc = np.random.uniform(0., 1., size=6)
    dc = np.reshape(dc, (3, 2))
    pc = np.random.uniform(0., 1., size=5)
    pc = np.reshape(pc, (len(pc), 1))
    print('g func:\n', g_func(dc, pc))



