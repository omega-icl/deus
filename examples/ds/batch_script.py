import numpy as np
from scipy.integrate import odeint
import time


'''
Batch Reactor (Kucherenko et al 2019):
Characterize probabilistic DS defined over d-space = {t_bach, T}
given uncertainty in model parameters space = {E1, E2, k01, k02}
and constraints in purity and profit rate.
'''


class BatchProcess:
    def __init__(self):
        # geometric parameters
        self.V = 1  # m3
        # process parameters
        self.t_batch = 0.0  # min
        self.T = 0.0        # K
        # model parameters
        self.E1_R = 0.0   # K
        self.E2_R = 0.0   # K
        self.k01 = 0.0  # 1/min
        self.k02 = 0.0  # 1/min
        # initial conditions
        self.ca0 = 2000.  # mol/m3
        self.cb0 = 0.0    # mol/m3
        self.cc0 = 0.0    # mol/m3
        # useful constants
        self.opt_profit = 160.  # $/min
        self.min_purity = 0.8  # mol B/mol mixture
        self.min_profit_frac = 0.8  # actual profit/optimal profit

    def set_process_parameters(self, d):
        self.t_batch, self.T = d

    def set_model_parameters(self, theta):
        self.E1_R, self.E2_R, self.k01, self.k02 = theta

    def eqns(self, x, t):
        ca, cb, cc = x

        k1 = self.k01 * np.exp(-self.E1_R / self.T)
        k2 = self.k02 * np.exp(-self.E2_R / self.T)

        r1 = k1 * ca*ca
        r2 = k2 * cb

        ddt_ca = -2.0 * self.V * r1
        ddt_cb = self.V * (r1 - r2)
        ddt_cc = self.V * r2

        return np.array([ddt_ca, ddt_cb, ddt_cc])

    def simulate(self, d, theta):
        assert isinstance(d, np.ndarray), "pass a numpy ndarray"
        assert isinstance(theta, np.ndarray), "pass a numpy ndarray"
        self.set_process_parameters(d.tolist())
        self.set_model_parameters(theta.tolist())
        ic = np.array([self.ca0, self.cb0, self.cc0])
        tspan = [0., self.t_batch]
        try:
            sim_solution = odeint(self.eqns, ic, tspan)
        except:
            sim_solution = None

        return sim_solution

    def cqas(self, sim_solution):
        if sim_solution is None:
            purity = -np.inf
            profit = -np.inf
        else:
            ca_final, cb_final, cc_final = sim_solution[-1, :]
            purity = cb_final / (ca_final + cb_final + cc_final)
            profit = (100.*cb_final - 20.*self.ca0)*self.V \
                / (self.t_batch + 30.)
        return purity, profit

    def constraints(self, cqas):
        purity, profit = cqas

        g1 = purity - self.min_purity
        g2 = profit - self.min_profit_frac * self.opt_profit
        return np.array([g1, g2])


def g_func(d_mat, theta_mat):
    process = BatchProcess()

    n_theta, d_theta = np.shape(theta_mat)
    d_g = 2

    g_mat_list = []
    for i, d_row in enumerate(d_mat):
        g_mat = np.ndarray((n_theta, d_g))
        for k, theta_row in enumerate(theta_mat):
            sim_sol = process.simulate(d_row, theta_row)
            cqas = process.cqas(sim_sol)
            g_vec = process.constraints(cqas)
            g_mat[k, :] = g_vec

        g_mat_list.append(g_mat)

    return g_mat_list


if __name__ == "__main__":
    _theta = np.array([[2500.2, 5000.1, 0.0641, 9938.1]])
    _d = np.array([[330., 285.]]*100)

    tic = time.time()
    _g = g_func(_d, _theta)
    toc = time.time()
    duration = toc - tic

    print("CPU seconds: %.8f" % duration)
    print("constraints:\n", _g)
