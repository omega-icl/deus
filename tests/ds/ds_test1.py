import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
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

    def g_func(self, d, p):
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
n_samples_p = 30
p_samples = np.random.normal(p_best, p_sdev, n_samples_p)
p_samples = [{'c': [p], 'w': 1.0/n_samples_p} for p in p_samples]

the_activity_form = {
    "activity_type": "dsc",

    "activity_settings": {
        "case_name": "DS_Test1",
        "case_path": os.getcwd(),
        "resume": False,
        "save_period": 5
    },

    "problem": {
        "user_script_filename": "ds_test2_user_script",
        "constraints_func_name": "g_func",
        "parameters_best_estimate": [p_best],
        "parameters_samples": p_samples,
        "target_reliability": 0.95,
        "design_variables": [
            {"d1": [-1.0, 1.0]},
            {"d2": [-1.0, 1.0]}
        ]
    },

    "solver": {
        "name": "dsc-ns",
        "settings": {
            "score_evaluation": {
                "method": "serial",
                "score_type": "sigmoid",  # "indicator",
                "constraints_func_ptr": the_model.g_func,
                # "constraints_func_ptr": None,
                "store_constraints": False
            },
            # "score_evaluation": {
            #     "method": "mppool",
            #     "score_type": "sigmoid",  # "indicator",
            #     "pool_size": -1,
            #     "store_constraints": False
            # },
            # "efp_evaluation": {
            #     "method": "serial",
            #     "constraints_func_ptr": the_model.g_func,
            #     # "constraints_func_ptr": None,
            #     "store_constraints": False,
            # },
            "efp_evaluation": {
                "method": "mppool",
                "pool_size": -1,
                "store_constraints": False
            },
            "phases_setup": {
                "initial": {
                    "nlive": 30,
                    "nproposals": 10
                },
                "deterministic": {
                    "skip": False
                },
                "nmvp_search": {
                    "skip": True
                },
                "probabilistic": {
                    "skip": False,
                    "nlive_change": {
                        "mode": "user_given",
                        "schedule": [
                            (.00, 320, 80),
                            (.25, 340, 80),
                            (.50, 360, 80),
                            (.75, 380, 60),
                            (.80, 400, 60)
                        ]
                    }
                }
            }
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": 10,  # This is overridden by points_schedule
                     "nproposals": 5,  # This is overriden by points_schedule
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

samples = output["solution"]["deterministic_phase"]["samples"]
coords_in_det_ds = np.empty((0, 2))
coords_out_det_ds = np.empty((0, 2))
score_type = \
    the_activity_form["solver"]["settings"]["score_evaluation"]["score_type"]
if score_type == "sigmoid":
    thrs = -2 * np.log(2)
elif score_type == "indicator":
    thrs = 1.0
for i, phi in enumerate(samples["phi"]):
    if phi >= thrs:
        coords_in_det_ds = np.append(
            coords_in_det_ds, [samples["coordinates"][i]], axis=0)
    else:
        coords_out_det_ds = np.append(
            coords_out_det_ds, [samples["coordinates"][i]], axis=0)

fig1 = plt.figure()
x = coords_in_det_ds[:, 0]
y = coords_in_det_ds[:, 1]
plt.scatter(x, y, s=10, c='g', alpha=0.5, label='inside nominal DS')

x = coords_out_det_ds[:, 0]
y = coords_out_det_ds[:, 1]
plt.scatter(x, y, s=10, c='b', alpha=0.25, label='outside nominal DS')
plt.legend()

samples = output["solution"]["probabilistic_phase"]["samples"]
coords_in_prob_ds = np.empty((0, 2))
coords_out_prob_ds = np.empty((0, 2))
thrs = the_activity_form["problem"]["target_reliability"]
for i, phi in enumerate(samples["phi"]):
    if phi >= thrs:
        coords_in_prob_ds = np.append(coords_in_prob_ds,
                                      [samples["coordinates"][i]], axis=0)
    else:
        coords_out_prob_ds = np.append(coords_out_prob_ds,
                                       [samples["coordinates"][i]], axis=0)

x = coords_in_prob_ds[:, 0]
y = coords_in_prob_ds[:, 1]
plt.scatter(x, y, s=10, c='r', alpha=1.0, label='inside target DS')

x = coords_out_prob_ds[:, 0]
y = coords_out_prob_ds[:, 1]
plt.scatter(x, y, s=10, c='k', alpha=0.75, label='outside target DS')
plt.legend()


fig2, ax = plt.subplots(1)
phase = "probabilistic_phase"
source = output["performance"][phase]

x = [item["iteration"] for item in source]
y1 = [item["cpu_time"]["proposing"]['main'] +
      item["cpu_time"]["proposing"]['topup']
      for item in source]
ax.plot(x, y1, 'b-o', label='proposals generation')

y2 = [item["cpu_time"]["evaluating"]['main'] +
      item["cpu_time"]["evaluating"]['topup']
      for item in source]
ax.plot(x, y2, 'r-o', label='phi evaluation')

y3 = np.array(y1) + np.array(y2)
ax.plot(x, y3, 'g--o', label='total')

ax.set_xlabel('iteration')
ax.set_ylabel('CPU seconds')
ax.grid()
ax.legend()


fig3, ax = plt.subplots(1)
phase = "probabilistic_phase"
source = output["performance"][phase]

x = [item["iteration"] for item in source]
y = [item["n_evals"]["phi"]["main"] +
     item["n_evals"]["phi"]["topup"]
     for item in source]
line1 = ax.plot(x, y, 'k-o', label='n proposals')

y = [item["n_replacements_done"] for item in source]
line2 = ax.plot(x, y, 'g-o', label='n replacements')

y = [item["n_evals"]["model"]["main"] +
     item["n_evals"]["model"]["topup"]
     for item in source]
ax2 = ax.twinx()
line3 = ax2.plot(x, y, 'b-o', label='n model evals')

ax2.set_ylabel('# model evaluations')
ax.set_xlabel('iteration')
ax.set_ylabel('# proposals|replacements')
ax.grid()
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=0)

plt.show()
