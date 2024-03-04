import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS
'''
Kucherenko et al 2019:
See 'batch_script.py' for a case description.
'''

# For nominal/deterministic phase
E1_nom = 2500.2
E2_nom = 5000.1
k01 = 0.0641
k02 = 9938.1
theta_nominal = [E1_nom, E2_nom, k01, k02]

# For probabilistic phase
theta_mean = np.array(theta_nominal)
variation_coeff = 0.01  # sdev_i/mean_i, i=1,...,4
theta_sdev = [variation_coeff*m for m in theta_mean]
theta_cov = np.zeros((4, 4))
np.fill_diagonal(theta_cov, [s*s for s in theta_sdev])
# Generate samples from model parameter distribution
np.random.seed(1989)
n_theta = 32
theta_samples = np.random.multivariate_normal(theta_mean, theta_cov, n_theta)
theta_samples = [{'c': p, 'w': 1.0/n_theta} for p in theta_samples]

the_activity_form = {
    "activity_type": "dsc",

    "activity_settings": {
        "case_name": "batch",
        "case_path": os.getcwd(),
        "resume": False,
        "save_period": 10
    },

    "problem": {
        "user_script_filename": "batch_script",
        "constraints_func_name": "g_func",
        "parameters_best_estimate": theta_nominal,
        "parameters_samples": theta_samples,
        "target_reliability": 0.75,
        "design_variables": [
            {"t_batch": [250., 350.]},
            {"T": [250., 350.]}
        ]
    },

    "solver": {
        "name": "dsc-ns",
        "settings": {
            # "score_evaluation": {
            #     "method": "serial",
            #     "score_type": "sigmoid",
            #     "constraints_func_ptr": None,
            #     "store_constraints": False
            # },
            "score_evaluation": {
                "method": "mppool",
                "score_type": "sigmoid",
                "pool_size": -1,
                "store_constraints": True
            },
            # "efp_evaluation": {
            #     "method": "serial",
            #     "constraints_func_ptr": None,
            #     "store_constraints": False
            # },
            "efp_evaluation": {
                "method": "mppool",
                "pool_size": -1,
                "store_constraints": False
            },
            "phases_setup": {
                "initial": {
                    "nlive": 200,
                    "nproposals": 60
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
                            (.00, 220, 80),
                            (.25, 240, 80),
                            (.50, 260, 80),
                            (.75, 280, 60),
                            (.80, 300, 60)
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
                     "f0": 0.05,
                     "alpha": 0.75,
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
threshold = the_activity_form["problem"]["target_reliability"]
for i, phi in enumerate(samples["phi"]):
    if phi >= threshold:
        inside_samples_coords = np.append(inside_samples_coords,
                                          [samples["coordinates"][i]], axis=0)
    else:
        outside_samples_coords = np.append(outside_samples_coords,
                                           [samples["coordinates"][i]], axis=0)

fig1 = plt.figure()
plt.xlabel(r'$\tau_{batch}$, min')
plt.ylabel(r'$T$, K')
plt.xlim([250, 350])
plt.ylim([250, 350])
x = inside_samples_coords[:, 0]
y = inside_samples_coords[:, 1]
plt.scatter(x, y, s=10, c='r', alpha=1.0, label='inside DS', rasterized=True)

x = outside_samples_coords[:, 0]
y = outside_samples_coords[:, 1]
plt.scatter(x, y, s=10, c='b', alpha=0.5, label='outside DS', rasterized=True)
plt.legend()
plt.grid()
plt.show()


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
