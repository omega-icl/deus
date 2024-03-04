import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS


'''
Parameter Estimation - Test 2:
See 'pe_test2_user_script.py' for a case description.
'''


an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "PeTest2",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "user_script_filename": "pe_test2_user_script",
        "log_prior_func_name": "the_log_prior",
        "log_lkhd_func_name": "the_log_lkhd",
        "parameters": [
            {"a": [-5, 5]},
            {"b": [-10, 10]}
        ]
    },

    "solver": {
        "name": "pe-ns",
        "settings": {
            # "log_lkhd_evaluation": {
            #     "method": "serial",
            #     "log_lkhd_ptr": None
            #     # "log_lkhd_ptr": the_log_lkhd
            # },
            "log_lkhd_evaluation": {
                "method": "mppool",
                "pool_size": -1
            },
            "stop_criteria": [
                {"contribution_to_evidence": 0.05}
            ]
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                     "nlive": 200,
                     "nproposals": 400,
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

the_deus = DEUS(an_activity_form)
t0 = time.time()
the_deus.solve()
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
plt.scatter(x, y, s=10, c='b', rasterized=True)

centre = [0, 2]
spread = [[1, 1.5],
          [1.5, 4]]
truth = np.random.multivariate_normal(mean=centre,
                                      cov=spread,
                                      size=1000)
x = truth[:, 0]
y = truth[:, 1]
plt.scatter(x, y, s=10, c='r', rasterized=True)


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

ax.set_xlabel('iteration')
ax.set_ylabel('CPU seconds')
ax.grid()
ax.legend()

plt.show()

