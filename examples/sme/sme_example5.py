import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
from scipy.integrate import odeint
import pickle
from scipy.stats import multivariate_normal
import time

from deus import DEUS


'''
Set-membership Parameter Estimation - Example 5:
    Dynamic bioreactor (anaerobic digester) parameter and state estimation
'''

# Setting up an example
class MyModel:
    def __init__(self):
        self.parameters = np.ndarray(5)
        '''mu1max, KS1, mu2max, KS2, KI2'''
        
        self.controls = np.ndarray(5)
        '''Din, CODin, VFAin, TICin, ALKin'''

    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def set_controls(self, controls):
        self.controls = controls

    def eqns(self, x, t):
        # parameters to estimate
        mu1max, KS1, mu2max, KS2, KI2 = self.parameters
        #mu1max  = 1.2   # /day
        #KS1     = 7.1   # g/L
        #mu2max  = 0.74  # /day
        #KS2     = 9.28  # mmol/l
        #KI2     = 256   # mmol/l
        
        # constant model parameters
        a       = 0.5      # mmol/l
        kLa     = 19.8e0   # /day
        k1      = 42.14e0  # g(COD)/g(VSS)
        k2      = 116.5e0  # mmol/g(VSS)
        k3      = 268.0e0  # mmol/g(VSS)
        k4      =  50.6e0  # mmol/g(VSS)
        k5      = 343.6e0  # mmol/g(VSS)
        k6      = 453.0e0  # mmol/g(VSS)
        kVFA    = 64e-3    # g(COD)/mmol
        KH      = 16e0     # mmol/L/atm
        PT      = 1e0      # atm
        
        # control inputs
        Din, CODin, VFAin, TICin, ALKin = self.controls
        
        # inputs definition
        D = Din
        S1in = CODin - VFAin * kVFA
        S2in = VFAin
        Zin = ALKin
        Cin = TICin
        
        # state variables
        X1, X2, S1, S2, Z, C = x
        
        # algebraic expressions
        mu1 = mu1max * S1 / (S1 + KS1)
        mu2 = mu2max * S2 / (S2 + KS2 + S2 * S2 / KI2)
        phi = C + S2 - Z + KH * PT + k6 * mu2 * X2 / kLa
        PC = (phi - np.sqrt(phi**2 - 4.0 * KH * PT * (C + S2 - Z))) / 2.0 / KH
        qC = kLa * (C + S2 - Z - KH * PC)
        
        return np.asarray([(mu1 - a * D) * X1, (mu2 - a * D) * X2,\
            D * (S1in - S1) - k1 * mu1 * X1,\
            D * (S2in - S2) + k2 * mu1 * X1 - k3 * mu2 * X2, D * (Zin - Z),\
            D * (Cin - C) - qC + k4 * mu1 * X1 + k5 * mu2 * X2])


def log_prior(p):
    # I am an unused dummy prior :D
    return -1.0e20


def log_lkhd(p):
    npar = np.size(p)
    
    err = np.array([1.0e-1, 1.0e-2, 1.0e-2])
    the_model = MyModel()
    # initial conditions
    X10 = 0.5   # g(VSS)/L
    X20 = 1.0   # g(VSS)/L
    S10 = 1.0   # g(COD)/L
    S20 = 5.0   # mmol/L
    C0  = 40.0  # mmol/L
    Z0  = 50.0  # mmol/L

    # inputs selection
    Dinvec   = np.array([0.25, 1.0, 1.0, 0.25])     # Dilution rate [/day]
    CODinvec = np.array([7.5, 7.5, 15.0, 7.5])      # Influent COD [g(COD)/L]
    VFAinvec = np.array([80.0, 80.0, 160.0, 80.0])  # Influent VFA [mmol/L]
    TICinvec = np.array([5.0, 5.0, 10.0, 5.0])      # Influent TIC [mmol/L]
    ALKinvec = np.array([50.0, 50.0, 100., 50.0])   # Influent ALK [mmol/L]

    # simulate dynamic model
    ptrue = np.array([1.2, 7.1, 0.74, 9.28, 256])
    the_model.set_parameters(ptrue)

    tspan = np.linspace(0.0, 1.0, 7)  # time grid
    x0 = [X10, X20, S10, S20, C0, Z0]  # initial conditions
    
    Y = np.zeros([np.size(Dinvec)*6, 3])
    xk = x0
    for i in range(np.size(Dinvec)):
        # set inputs
        the_model.set_controls([Dinvec[i], CODinvec[i],\
            VFAinvec[i], TICinvec[i], ALKinvec[i]])
        # integrate
        solution = np.array(odeint(the_model.eqns, xk, tspan))
        xk = solution[-1, :]        
        Y[i*6:(i*6+6), :] = solution[1:7, [2, 3, 5]]
        
        ym = np.transpose([np.round(Y[:, 0]/2.0/err[0])*2.0*err[0],
              np.round(Y[:, 1]/2.0/err[1])*2.0*err[1],
              np.round(Y[:, 2]/2.0/err[2])*2.0*err[2]])

        nexp, ny = np.shape(ym)        
        inv_covariance = np.kron(np.eye(nexp), np.eye(ny)/(err/3.0)**2)
           
    the_model.set_parameters(p[0:5])
        
    Y = np.zeros([np.size(Dinvec)*6, 3]);
    xk = p[5:11]
        
    for i in range(np.size(Dinvec)):
        # set inputs
        the_model.set_controls([Dinvec[i], CODinvec[i],\
            VFAinvec[i], TICinvec[i], ALKinvec[i]])
        # integrate
        solution = np.array(odeint(the_model.eqns, xk, tspan))
        xk = solution[-1, :]        
        Y[i*6:(i*6+6), :] = solution[1:7, [2, 3, 5]]
        
    e1 = Y[:, 0] - ym[:, 0]
    e2 = Y[:, 1] - ym[:, 1]
    e3 = Y[:, 2] - ym[:, 2]
    if np.all(-err[0]*np.ones(nexp) <= e1) and\
        np.all(e1 <= err[0]*np.ones(nexp)) and\
        np.all(-err[1]*np.ones(nexp) <= e2) and\
        np.all(e2 <= err[1]*np.ones(nexp)) and\
        np.all(-err[2]*np.ones(nexp) <= e3) and\
        np.all(e3 <= err[2]*np.ones(nexp)):
        f = 0
    else:
        f = -1/2*(np.reshape(np.transpose(Y), nexp*ny, 1) -\
            np.reshape(np.transpose(ym), nexp*ny, 1))@inv_covariance@\
                np.transpose((np.reshape(np.transpose(Y), nexp*ny, 1) -\
                    np.reshape(np.transpose(ym), nexp*ny, 1)))
    return f

# Setting up options (including search range in "problem":"parameters")
an_activity_form = {
    "activity_type": "pe",

    "activity_settings": {
        "case_name": "SmeExample5",
        "case_path": getcwd(),
        "resume": False,
        "save_period": 1
    },

    "problem": {
        "goal": "posterior",    # TODO errors in problem definition, all checks passed for PE and DS
        "log_pi": log_prior,
        "log_l": log_lkhd,
        "parameters": [
            {"mu1max": [0.5, 1.5]},
            {"KS1": [5.5, 8.0]},
            {"mu2max": [0.735, 0.745]},
            {"KS2": [9.1, 9.35]},
            {"KI2": [250.0, 265.0]},
            {"X10": [0.3, 0.7]},
            {"X20": [0.8, 1.2]},
            {"S10": [0.8, 1.2]},
            {"S20": [4.0, 6.0]},
            {"C0": [38.0, 42.0]},
            {"Z0": [48.0, 52.0]}
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
samples_coords = np.empty((0, 11))
samples_weights = np.empty(0)
for i, sample in enumerate(samples["coordinates"]):
    if samples["log_l"][i] == 0:
        samples_coords = np.append(samples_coords, np.array([samples["coordinates"][i]]), axis=0)
    samples_weights = np.append(samples_weights, [weights[i]],
                                axis=0)


mu1max = samples_coords[:, 0]
KS1 = samples_coords[:, 1]
mu2max = samples_coords[:, 2]
KS2 = samples_coords[:, 3]
KI2 = samples_coords[:, 4]

fig1 = plt.figure()
plt.scatter(KS1, mu1max, s=10, c='b')

fig2 = plt.figure()
plt.scatter(mu2max, KS1, s=10, c='b')

fig3 = plt.figure()
plt.scatter(KS2, mu2max, s=10, c='b')

fig4 = plt.figure()
plt.scatter(KI2, KS2, s=10, c='b')

X10 = samples_coords[:, 5]
X20 = samples_coords[:, 6]
S10 = samples_coords[:, 7]
S20 = samples_coords[:, 8]
C0 = samples_coords[:, 9]
Z0 = samples_coords[:, 10]

fig5 = plt.figure()
plt.scatter(X10, S10, s=10, c='b')

fig6 = plt.figure()
plt.scatter(X20, X10, s=10, c='b')

fig7 = plt.figure()
plt.scatter(X20, S20, s=10, c='b')

fig8 = plt.figure()
plt.scatter(S20, C0, s=10, c='b')

fig9 = plt.figure()
plt.scatter(C0, Z0, s=10, c='b')

plt.show()
