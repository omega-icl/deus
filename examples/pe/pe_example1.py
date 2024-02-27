import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS
'''
Parameter Estimation - Example 1:
Let the 2D Rosenbrock function be the likelihood function
'''


def log_prior(p):
    # I am an unused dummy prior :D
    return -300


def log_lkhd(p):
    n_dims = 2
    rosenbrock = 0
    for j in range(n_dims-1):
        rosenbrock += 100.0*(p[j+1] - p[j]**2)**2 + (1.0 - p[j])**2
    log_l = -rosenbrock
    return log_l     # TODO this generates an error (100 points evaluated, results in 2 fn values. )


an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "PE_Example1",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "user_script_filename": "pe_test1_user_script",
        "log_prior_func_name": "log_prior",
        "log_lkhd_func_name": "log_lkhd",
        "parameters": [
            {"theta1": [-10.0, 10.0]},
            {"theta2": [-10.0, 10.0]}
        ]
    },

    "solver": {
        "name": "pe-ns",
        "settings": {
            "log_lkhd_evaluation": {
                "method": "serial",
                "log_lkhd_ptr": log_lkhd
            },
            "stop_criteria": [
                {"contribution_to_evidence": 0.05}
            ]
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                    "nlive": 100,
                    "nreplacements": 75,
                    "prng_seed": 1989,
                    "f0": 0.3,
                    "alpha": 0.2,
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

the_duu = DEUS(an_activity_form)
t0 = time.time()
the_duu.solve()
cpu_time = time.time() - t0
print('CPU seconds', cpu_time)

cs_path = an_activity_form["activity_settings"]["case_path"]
cs_name = an_activity_form["activity_settings"]["case_name"]

with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb')\
        as file:
    output = pickle.load(file)

log_z_mean = output["solution"]["log_z"]["hat"]
log_z_sdev = output["solution"]["log_z"]["sdev"]
h = output["solution"]["post_prior_kldiv"]
print('log Z =', log_z_mean, '+/-', log_z_sdev)
print('H =', h)


samples = output["solution"]["samples"]
weights = output["solution"]["samples"]["weights"]
samples_coords = np.empty((0, 2))
samples_weights = np.empty(0)
for i, sample in enumerate(samples["coordinates"]):
    samples_coords = np.append(samples_coords, [sample],
                               axis=0)
    samples_weights = np.append(samples_weights, [weights[i]],
                                axis=0)

fig1 = plt.figure()
x = samples_coords[:, 0]
y = samples_coords[:, 1]
plt.scatter(x, y, s=10, c='b')

fig2 = plt.figure()
x = np.arange(len(samples_weights))
y = samples_weights
plt.plot(x, y, c='g')


fig3, ax = plt.subplots(1)
x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["proposals"]
     for item in output["performance"]]
ax.plot(x, y, 'b-', label='proposals generation')

x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["lkhd_evals"]
     for item in output["performance"]]
ax.plot(x, y, 'r-', label='likelihood evaluations')

x = [item["iteration"] for item in output["performance"]]
y = [item["cpu_secs"]["total"] for item in output["performance"]]
ax.plot(x, y, 'g--', label='total')

ax.set_ylabel('CPU seconds')
ax.grid()
ax.legend()


plt.show()
