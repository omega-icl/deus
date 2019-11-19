import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS


'''
Design Space - Test 1:
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


class ModelA:
    def __init__(self):
        pass

    def s(self, d, p):
        time.sleep(0.0001)
        d1, d2 = d
        p1 = p[0]
        x1 = p1*d1**2 + d2
        return x1

    def g_cat0(self, d, p):
        s = self.s(d, p)

        g1 = s - 0.2
        g2 = 0.75 - s
        ans = np.array([g1, g2])
        return ans

    def g_func_cat2(self, d, p):
        '''
        :param d: 2d array, each row is a design vector
        :param p: 2d array, each row is a parameter vector
        :return: list of 2d array, j^th row in i^th item is g(d_i, p_j)
        '''

        n_p, p_dims = np.shape(p)
        g_dims = 2

        g_list = []
        for i, d_vec in enumerate(d):
            g_mat = np.empty((n_p, g_dims))
            for j, p_vec in enumerate(p):
                s = self.s(d_vec, p_vec)
                g1 = s - 0.2
                g2 = 0.75 - s
                g_mat[j, :] = np.array([g1, g2])

            g_list.append(g_mat)
        return g_list


the_model = ModelA()

p_best = 1.0
p_sdev = np.sqrt(0.3)

np.random.seed(1)
n_samples_p = 100
p_samples = np.random.normal(p_best, p_sdev, n_samples_p)
p_samples = [{'c': [p], 'w': 1.0/n_samples_p} for p in p_samples]

the_activity_form = {
    "activity_type": "ds",

    "activity_settings": {
        "case_name": "DS_Test1",
        "case_path": os.getcwd(),
        "resume": False,
        "save_period": 10
    },

    "problem": {
        "user_script_filename": "ds_test2_user_script",
        "constraints_func_name": "g_func_cat2",
        "parameters_best_estimate": [p_best],
        "parameters_samples": p_samples,
        "target_reliability": 0.95,
        "design_variables": [
            {"d1": [-1.0, 1.0]},
            {"d2": [-1.0, 1.0]}
        ]
    },

    "solver": {
        "name": "ds-ns",
        "settings": {
            # "score_evaluation": {
            #     "method": "serial",
            #     "constraints_func_ptr": the_model.g_func_cat2,
            #     # "constraints_func_ptr": None,
            #     "store_constraints": False
            # },
            "score_evaluation": {
                "method": "mppool",
                "pool_size": -1,
                "constraints_func_ptr": None,
                "store_constraints": False
            },
            # "efp_evaluation": {
            #     "method": "serial",
            #     "constraints_func_ptr": the_model.g_func_cat2,
            #     # "constraints_func_ptr": None,
            #     "store_constraints": False,
            #     "acceleration": False
            # },
            "efp_evaluation": {
                "method": "mppool",
                "pool_size": -1,
                "constraints_func_ptr": None,
                "store_constraints": False,
                "acceleration": False
            },
            "points_schedule": [
                (.0, 30, 30),
                (.01, 100, 40),
                (.1, 150, 40),
                (.5, 200, 40),
                (.7, 250, 40),
                (.8, 300, 40)
            ],
            "stop_criteria": [
                {"inside_fraction": 1.0}
            ]
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": 10,  # This is overridden by points_schedule
                     "nreplacements": 5,  # This is overriden by points_schedule
                     "prng_seed": 1989,
                     "f0": 0.1,
                     "alpha": 0.3,
                     "stop_criteria": [
                         {"max_iterations": 100000}
                     ],
                     "debug_level": 0,
                     "monitor_performance": False
                 },
                "algorithms": {
                    "replacement": {
                        "sampling": {
                            "algorithm": "suob-ellipsoid"
                        }
                    }
                }
            }
        }
    }
}

the_deus = DEUS(the_activity_form)
t0 = time.time()
the_deus.solve()
cpu_secs = time.time() - t0
print('CPU seconds', cpu_secs)

cs_path = the_activity_form["activity_settings"]["case_path"]
cs_name = the_activity_form["activity_settings"]["case_name"]

with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb') \
        as file:
    output = pickle.load(file)

samples = output["solution"]["probabilistic_phase"]["samples"]
inside_samples_coords = np.empty((0, 2))
outside_samples_coords = np.empty((0, 2))
for i, phi in enumerate(samples["phi"]):
    if phi >= 0.95:
        inside_samples_coords = np.append(inside_samples_coords,
                                          [samples["coordinates"][i]], axis=0)
    else:
        outside_samples_coords = np.append(outside_samples_coords,
                                           [samples["coordinates"][i]], axis=0)

fig1 = plt.figure()
x = inside_samples_coords[:, 0]
y = inside_samples_coords[:, 1]
plt.scatter(x, y, s=10, c='r', alpha=1.0, label='inside')

x = outside_samples_coords[:, 0]
y = outside_samples_coords[:, 1]
plt.scatter(x, y, s=10, c='b', alpha=0.5, label='outside')
plt.legend()


fig2, ax = plt.subplots(1)
x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["proposals"]
     for item in output["performance"]]
ax.plot(x, y, 'b-', label='proposals generation')

x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["phi_evals"]
     for item in output["performance"]]
ax.plot(x, y, 'r-', label='phi evaluations')

x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["total"] for item in output["performance"]]
ax.plot(x, y, 'g--', label='total')

ax.set_ylabel('CPU seconds')
ax.grid()
ax.legend()


fig3, ax = plt.subplots(1)
x = [item["iteration"]
     for item in output["performance"]]
y = [item["n_proposals"]
     for item in output["performance"]]
line1 = ax.plot(x, y, 'k-', label='n proposals')

x = [item["iteration"]
     for item in output["performance"]]
y = [item["n_replacements"]
     for item in output["performance"]]
line2 = ax.plot(x, y, 'g-', label='n replacements')

x = [item["iteration"]
     for item in output["performance"]]
y = [item["n_model_evals"]
     for item in output["performance"]]
ax2 = ax.twinx()
line3 = ax2.plot(x, y, 'b-', label='n model evals')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=0)

plt.show()
