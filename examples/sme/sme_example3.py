import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from scipy.integrate import odeint
import pickle
from scipy.stats import multivariate_normal
import time

from duu import DUU


'''
Set-membership Parameter Estimation - Example 3:
    Dynamic 3D estimation under lack of identifiability
'''

# Setting up an example
class MyModel:
    def __init__(self):
        self.parameters = np.ndarray(3)
        '''k1, k2, k3'''        

    def set_parameters(self, parameters):
        self.parameters = parameters

    def eqns(self, x, t):
        # parameters to estimate
        k1, k2, k3 = self.parameters

        # state variables
        x1, x2 = x

        dx1 = -(k1+k3)*x1 + k2*x2
        dx2 = k1*x1 - k2*x2
        return np.asarray([dx1, dx2])


def log_prior(p):
    # I am an unused dummy prior :D
    return -1.0e20


def log_lkhd(p):    
    err = 0.005
    the_model = MyModel()
    ptrue = [0.6, 0.15, 0.35]
    the_model.set_parameters(ptrue)
    
    tspan = np.linspace(0.0, 15.0, 16)  # time grid
    x0 = [1.0, 0.0]  # initial conditions

    solution = odeint(the_model.eqns, x0, tspan)
    ym = np.round(100.0*np.asarray(solution[1:, 1]))/100.0
    ny = np.size(ym)
    
    inv_covariance = np.eye(ny)/(err/3.0)**2
            
    the_model.set_parameters(p)
    solution = odeint(the_model.eqns, x0, tspan)
    y = np.asarray(solution[1:, 1])
    
    e = y - ym
    if np.all(-err*np.ones(ny) <= e) and np.all(e <= err*np.ones(ny)):
        f = 0
    else:
        f = -1/2*e@inv_covariance@np.transpose(e)
    return f

# Setting up options (including search range in "problem":"parameters")
an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "SmeExample3",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "goal": "posterior",
        "log_pi": log_prior,
        "log_l": log_lkhd,
        "parameters": [
            {"theta1": [0.0, 1.0]},
            {"theta2": [0.0, 1.0]},
            {"theta3": [0.0, 1.0]}
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
samples_coords = np.empty((0, 3))
samples_weights = np.empty(0)
for i, sample in enumerate(samples["coordinates"]):
    if samples["log_l"][i] == 0:
        samples_coords = np.append(samples_coords, np.array([samples["coordinates"][i]]), axis=0)
    samples_weights = np.append(samples_weights, [weights[i]],
                                axis=0)

fig1 = plt.figure()
k1 = samples_coords[:, 0]
k2 = samples_coords[:, 1]
k3 = samples_coords[:, 2]
plt.scatter(k1, k2, s=10, c='b')

fig2 = plt.figure()
plt.scatter(k2, k3, s=10, c='b')

plt.show()
