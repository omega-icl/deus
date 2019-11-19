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


def g_func_cat0(d, p):
    '''
    :param d: 1d array
    :param p: 1d array
    :return: 1d array containing constraints
    '''

    a_model = Model()
    s = a_model.s(d, p)
    g1 = s - 0.2
    g2 = 0.75 - s
    g = np.array([g1, g2])
    return g


def g_func_cat1(d, p):
    '''
    :param d: 1d array
    :param p: 2d array, each row is a parameter vector
    :return: 2d array, i^th row represents the constraints for (d, p[i, :])
    '''

    n_p, p_dims = np.shape(p)
    g_dims = 2

    a_model = Model()
    g = np.empty((n_p, g_dims))
    for i, p_vec in enumerate(p):
        s = a_model.s(d, p_vec)
        g1 = s - 0.2
        g2 = 0.75 - s
        g[i, :] = np.array([g1, g2])
    return g


def g_func_cat2(d, p):
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

    dc0 = np.array([0.5, 0.8])
    pc0 = np.array([1.0])
    print('g func cat 0:\n', g_func_cat0(dc0, pc0))

    dc1 = np.array([0.5, 0.8])
    pc1 = np.random.uniform(0., 1., size=10)
    pc1 = np.reshape(pc1, (len(pc1), 1))
    print('g func cat 1:\n', g_func_cat1(dc1, pc1))

    dc2 = np.random.uniform(0., 1., size=10)
    dc2 = np.reshape(dc2, (5, 2))
    pc2 = np.random.uniform(0., 1., size=10)
    pc2 = np.reshape(pc2, (len(pc2), 1))
    print('g func cat 2:\n', g_func_cat2(dc2, pc2))



