import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS


'''
Set-membership Parameter Estimation - Example 2:
    Static nonlinear 2D estimation
'''

# Setting up an example
def log_prior(p):
    # I am an unused dummy prior :D
    return -1.0e20


def log_lkhd(p):    
    x = np.linspace(0, 1, 11)
    p1 = 1.0; p2 = 1.0
    y = p1*np.exp(p2*x)
    ny = np.size(y)
    err = 1.0
    
    inv_covariance = np.eye(ny)*3.0**2
    
    e = y - p[0]*np.exp(p[1]*x)
    if np.all(-err*np.ones(ny) <= e) and np.all(e<= err*np.ones(ny)):
        f = 0
    else:
        f = -1/2*e@inv_covariance@np.transpose(e)
    return f

# Setting up options (including search range in "problem":"parameters")
an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "SmeExample2",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "goal": "posterior",   # TODO errors in problem definition, all checks passed for PE and DS
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
                    "nlive": 600,
                    "nreplacements": 300,
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
samples_weights = np.empty(0)
for i, sample in enumerate(samples["coordinates"]):
    if samples["log_l"][i] == 0:
        samples_coords = np.append(samples_coords, np.array([samples["coordinates"][i]]), axis=0)
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
