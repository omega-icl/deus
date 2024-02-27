import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
import pickle
from scipy.stats import multivariate_normal
import time
from casadi import *       # TODO casadi must be installed, or this will fail

from deus import DEUS


'''
Set-membership Parameter Estimation - Example 4:
    Dynamic 3D estimation under lack of identifiability (ODE integration using sundials)
Note: CasADi package is required to run this example. (Example 3 uses odeint integrator.)
'''

# Setting up an example
# Formulate the ODE
x=SX.sym('x', 2)
k=SX.sym('k', 3)
f=vertcat(-(k[0]+k[2])*x[0] + k[1]*x[1], k[0]*x[0] - k[1]*x[1])
dae=dict(x=x,p=k,ode=f)

# Create solver instance
options=dict(t0=0, tf=1)
F=integrator('F','cvodes',dae,options)

ptrue = [0.6, 0.15, 0.35]
x0 = [1, 0]  # initial conditions

y = np.zeros([15, 1])
# Solve the problem
for i in range(1, 15):
    r=F(x0=x0, p=[0.6, 0.15, 0.35])
    x0 = r['xf']
    y[i-1] = x0[1]

ym = np.round(100.0*np.asarray(y))/100.0
ny = np.size(ym)

err = 0.005
inv_covariance = np.eye(ny)/(err/3.0)**2
    
def log_prior(p):
    # I am an unused dummy prior :D
    return -1.0e20

def log_lkhd(p):    
    x0 = [1, 0]
    y = np.zeros([15, 1])
    # Solve the problem
    for i in range(1, 15):
        r=F(x0=x0, p=p)
        x0 = r['xf']
        y[i-1] = x0[1]
        
    y = np.asarray(y)
    
    e = np.transpose(y - ym)
    if np.all(-err*np.ones(ny) <= e) and np.all(e <= err*np.ones(ny)):
        f = 0
    else:
        f = -1/2*e@inv_covariance@np.transpose(e)
    return f

# Setting up options (including search range in "problem":"parameters")
an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "SmeExample4",
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
