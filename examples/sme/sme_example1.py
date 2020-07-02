import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time

from duu import DUU


'''
Set-membership Parameter Estimation - Example 1: nD box
    Sample from a box [0, 2] x [0, 2] x ... x [0, 2]
'''

# Setting up an example
def log_prior(p):
    # I am an unused dummy prior :D
    return -1.0e20


def log_lkhd(p):
    npar = np.size(p)    
    inv_covariance = np.eye(npar)*3.0**2
    
    if np.all(0.0*np.ones(npar) <= p) and np.all(p <= 2.0*np.ones(npar)):
        f = 0
    else:
        f = -1/2*(p-1.0*np.ones(npar))@inv_covariance@np.transpose(p-1.0*np.ones(npar))        
    return f

# set number of dimensions
npar = 3
# set search domain breadth
a = 10**2
params = []
for ip in range(npar):
    params.append({"theta"+str(ip+1): [-a, a]})

# Setting up options (including search range in "problem":"parameters")
an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "SmeExample1",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "goal": "posterior",
        "log_pi": log_prior,
        "log_l": log_lkhd,
        "parameters": params
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
for ip in range(len(output['performance'])):
    n_eval += output['performance'][ip]["n_proposals"]
    
print('Number of evaluations: ', n_eval)

log_z_mean = output["solution"]["log_z"]["hat"]
log_z_sdev = output["solution"]["log_z"]["sdev"]
h = output["solution"]["post_prior_kldiv"]
print('log Z =', log_z_mean, '+/-', log_z_sdev)
print('H =', h)
        
samples = output["solution"]["samples"]
samples_coords = np.empty((0, npar))
for i, sample in enumerate(samples["coordinates"]):
    if samples["log_l"][i] == 0:
        samples_coords = np.append(samples_coords, np.array([samples["coordinates"][i]]), axis=0)


fig1 = plt.figure()
x1 = samples_coords[:, 0]
x2 = samples_coords[:, 1]
plt.scatter(x1, x2, s=10, c='b')

plt.show()
