import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from duu import DUU


'''
Set-membership Parameter Estimation - Test 1: 2D box
    Sample from a box [0, 2] x [0, 2]
'''

# Setting up an example
def log_prior(p):
    # I am an unused dummy prior
    return -1.0e20


def log_lkhd(p):
    # this test uses a simple indicator function
    if np.all(0.0*np.ones(2) <= p) and np.all(p <= 2.0*np.ones(2)):
        f = 0
    else:
        f = -1.0e20
    return f

# Setting up options (including search range in "problem":"parameters")
an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "SmeTest2",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "goal": "posterior",
        "log_pi": log_prior,
        "log_l": log_lkhd,
        "parameters": [
            {"theta1": [-10.0, 10.0]},
            {"theta2": [-10.0, 10.0]}
        ]
    },

    "solver": {
        "name": "pe-ns",
        "settings": {
            "parallel": "no",
            "stop_criteria": [
                {"contribution_to_evidence": 0.10}
            ]
        },
        "algorithms": {
            "sampling": {
                "algorithm": "mc_sampling-ns_global",
                "settings": {
                    "nlive": 300,
                    "nreplacements": 150,
                    "prng_seed": 1989,
                    "f0": 0.3,
                    "alpha": 0.2,
                    "stop_criteria": [
                        {"max_iterations": 10000}
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

# Running and processing
the_duu = DUU(an_activity_form)
t0 = time.time()
the_duu.solve()
cpu_time = time.time() - t0
print('CPU seconds', cpu_time)

cs_path = an_activity_form["activity_settings"]["case_path"]
cs_name = an_activity_form["activity_settings"]["case_name"]

with open(cs_path + '/' + cs_name + '/' + 'output.pkl', 'rb')\
        as file:
    output = pickle.load(file)

n_eval = an_activity_form["solver"]["algorithms"]["sampling"]["settings"]["nlive"];
for i in range(len(output['performance'])):
    n_eval += output['performance'][i]["n_proposals"]
    
print('Number of evaluations: ', n_eval)

log_z_mean = output["solution"]["log_z"]["hat"]
log_z_sdev = output["solution"]["log_z"]["sdev"]
h = output["solution"]["post_prior_kldiv"]
print('log Z =', log_z_mean, '+/-', log_z_sdev)
print('H =', h)

samples = output["solution"]["samples"]
weights = output["solution"]["samples"]["weights"]
samples_coords = np.empty((0, 2))
samples_coords_in = np.empty((0, 2))
samples_weights = np.empty(0)
for i, sample in enumerate(samples["coordinates"]):
    samples_coords = np.append(samples_coords, [sample],
                               axis=0)
    samples_weights = np.append(samples_weights, [weights[i]],
                                axis=0)
    if samples["log_l"][i] == 0:
        samples_coords_in = np.append(samples_coords_in, np.array([samples["coordinates"][i]]), axis=0)


fig1 = plt.figure()
x = samples_coords[:, 0]
y = samples_coords[:, 1]
plt.scatter(x, y, s=5, c='r')
x = samples_coords_in[:, 0]
y = samples_coords_in[:, 1]
plt.scatter(x, y, s=10, c='b')

fig2 = plt.figure()
plt.scatter(x, y, s=5, c='r')
x = samples_coords_in[:, 0]
y = samples_coords_in[:, 1]
plt.scatter(x, y, s=10, c='b')

fig3 = plt.figure()
x = np.arange(len(samples_weights))
y = samples_weights
plt.plot(x, y, c='g')

fig4, ax = plt.subplots(1)
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
