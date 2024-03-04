import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS


'''
Set Membership Estimation - Test 1:
Characterize the parameters set which guarantees that the errors
are within the given bounds i.e., e_i \in [-e_bar,i, +e_bar,i] 
for i=1,...,N.
'''


def the_errors_func(p):
    p_num, p_dim = np.shape(p)

    y_msred = np.array([0.0, 0.0])

    errors_mat = np.ndarray((p_num, 2))
    for i, p_vec in enumerate(p):
        y_model = p_vec
        errors_mat[i, :] = y_model - y_msred

    answer = errors_mat
    return answer


the_activity_form = {
    "activity_type": "sme",

    "activity_settings": {
        "case_path": getcwd(),
        "case_name": "SmeTest1",
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "user_script_filename": "sme_test1_user_script",
        "errors_func_name": "the_errors_func",
        "errors_bound": [1., 1.],
        "parameters": [
            {"p1": [-10, 10]},
            {"p2": [-10, 10]}
        ]
    },

    "solver": {
        "name": "sme-ns",
        "settings": {
            "spread_to_error_bound": 0.3333,
            "errors_evaluation": {
                "method": "serial",
                "errors_func_ptr": the_errors_func
                # "errors_func_ptr": None
            }
            # "errors_evaluation": {
            #     "method": "mppool",
            #     "pool_size": -1,
            # }
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": 200,
                     "nproposals": 100,
                     "prng_seed": 1989,
                     "f0": 0.1,
                     "alpha": 0.05,
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
cpu_time = time.time() - t0
print('CPU seconds', cpu_time)

cs_path = the_activity_form["activity_settings"]["case_path"]
cs_name = the_activity_form["activity_settings"]["case_name"]

with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb')\
        as file:
    output = pickle.load(file)

samples = output["solution"]["search_phase"]["samples"]
coords_in_set = np.empty((0, 2))
coords_out_set = np.empty((0, 2))

thrs = 0.0

for i, phi in enumerate(samples["phi"]):
    if phi >= thrs:
        coords_in_set = np.append(
            coords_in_set, [samples["coordinates"][i]], axis=0)
    else:
        coords_out_set = np.append(
            coords_out_set, [samples["coordinates"][i]], axis=0)

fig1 = plt.figure()
x = coords_in_set[:, 0]
y = coords_in_set[:, 1]
plt.scatter(x, y, s=10, c='g', alpha=1.0, label='inside target set')

x = coords_out_set[:, 0]
y = coords_out_set[:, 1]
plt.scatter(x, y, s=10, c='r', alpha=0.75, label='outside target set')
plt.legend()


fig2, ax = plt.subplots(1)
phase = "search_phase"
source = output["performance"][phase]

x = [item["iteration"] for item in source]
y1 = [item["cpu_time"]["proposing"] for item in source]
ax.plot(x, y1, 'b-o', label='proposals generation')

y2 = [item["cpu_time"]["evaluating"] for item in source]
ax.plot(x, y2, 'r-o', label='phi evaluation')

y3 = np.array(y1) + np.array(y2)
ax.plot(x, y3, 'g--o', label='total')

ax.set_xlabel('iteration')
ax.set_ylabel('CPU seconds')
ax.grid()
ax.legend()


fig3, ax = plt.subplots(1)
phase = "search_phase"
source = output["performance"][phase]

x = [item["iteration"] for item in source]
y = [item["n_evals"]["phi"] for item in source]
line1 = ax.plot(x, y, 'k-o', label='n proposals')

y = [item["n_replacements_done"] for item in source]
line2 = ax.plot(x, y, 'g-o', label='n replacements')

y = [item["n_evals"]["model"] for item in source]
ax2 = ax.twinx()
line3 = ax2.plot(x, y, 'b-o', label='n model evals')

ax2.set_ylabel('# model evaluations')
ax.set_xlabel('iteration')
ax.set_ylabel('# proposals | replacements')
ax.grid()
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc=0)

plt.show()