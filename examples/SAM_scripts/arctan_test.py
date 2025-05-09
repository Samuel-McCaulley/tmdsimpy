import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tmdsimpy.nlforces.arctangent_stiffness import *
from tmdsimpy.nlforces.iwan4_element import *
from tmdsimpy.nlforces.bouc_wen import *
from tmdsimpy.utils.harmonic import *
import matplotlib.pyplot as plt
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.vibration_system import VibrationSystem
from sdof_iwan_epmc import sdof_uwxa_full
import tmdsimpy.nlutils as nlutils
from scipy import io as sio
import time
import cProfile
import pstats
from scipy.special import betainc


m = 1
c = 0.005
k = 1e12

lparams = [4.957259556938594, 11.737845846995377, -0.023785113541144, -2.953624228947613, 16.933461451698992]
#lparams = [5.530289365884571, 15.859238733093676, -0.999870602783605, -5.746331062210725, 16.888652840667923] #Best Freq

lpsci = [1, 1, 0, 1, 1]
iwan_parameters = [10 ** lparams[i] if lpsci[i] == 1 else lparams[i] for i in range(len(lparams))]



kt = iwan_parameters[1] * 0.000385483218541205 
Fs = iwan_parameters[0] * 0.000385483218541205 # N, Match Jenkins
chi = iwan_parameters[2]  # Have a more full hysteresis loop than chi=0.0
beta = iwan_parameters[3]  # Smooth Transition

config = {
    'Astart': -10,
    'Aend': -2}

test_iwan = Iwan4Force(np.array([[1]]), np.array([[1]]), kt, Fs, chi, beta)

Uwxa_full = sdof_uwxa_full(m, c, k, test_iwan, config)

X0norm = np.zeros((Uwxa_full.shape[0]))
for line in range(Uwxa_full.shape[0]):
    X0norm[line] = np.linalg.norm(Uwxa_full[line, :-3]*10**Uwxa_full[line, -1])

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.title("Reference Backbone")
plt.show()

plt.plot(Uwxa_full[:, -1], np.log10(X0norm))
plt.title("Inertial effect")
plt.ylabel("X0 2-norm")
plt.xlabel("Amplitude")
plt.show()
beginning_amplitude = 10**Uwxa_full[0, -1] #Should be around astart

h = np.array([0, 1, 2, 3, ])
cst = np.ones((1 << 11, hutils.Nhc(h)))
Q = np.array([[1]])
T = np.array([[1]])

#Compute center-shift
max_omega = Uwxa_full[:, -3].max()
min_omega = Uwxa_full[:, -3].min()
center_omega = 1/2*(max_omega + min_omega)
omega_index = np.argmin(np.abs(Uwxa_full[:, -3] - center_omega))

#%%

"""
Solve for Parameters
"""

phi_max = Fs*(1 + beta)/(kt * (beta + (chi + 1)/(chi + 2)))
Kt_true = kt*(1 - (beginning_amplitude/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))


U0_iwan = Uwxa_full[0, :-3]*10**Uwxa_full[0, -1]
w0_iwan = Uwxa_full[0, -3]
times, disp, forces_iwan = nlutils.hysteresis_loop(1 << 10, h, np.atleast_2d(U0_iwan).T, w0_iwan, 'iwan', [kt, Fs, chi, beta])

ks_iwan_initial = (forces_iwan[np.argmax(disp)] - forces_iwan[np.argmin(disp)])/(max(disp) - min(disp))
print(f"Initial Iwan Tangent Stiffness: {ks_iwan_initial}")



#s = Uwxa_full[omega_index, -1]
#b = ArctangentStiffness.solve_for_b(Q, T, 10**Uwxa_full[52, -1], Uwxa_full[52, -2], Uwxa_full[52, -3], kt, s, m, c, verbose=False)
#wf, b, s = ArctangentStiffness.incomplete_slip_parameters(Uwxa_full[:, -1], Uwxa_full[:, -3], Uwxa_full[:, -2], True, True)
wf, b, s = ArctangentStiffness.incomplete_slip_parameters(np.log10(X0norm), Uwxa_full[:, -3], Uwxa_full[:, -2], True, True)
'''
MAJOR TEST 4/6/2025: Forcing initial kt to be matching iwan.
Adjustment 4/7/2025: Adjusting this value to reflect difference between 10^-infty and 10^start
'''
adjusted_kt_for_arc = ks_iwan_initial * 2 * (1 - 2/np.pi * np.arctan(b * np.log(beginning_amplitude) - s))**-1
arctangent_force = ArctangentStiffness(np.array([[1]]), np.array([[1]]), adjusted_kt_for_arc, b, s)



#%% Testing Validity: Does the sytem converge? I have no fucking clue, but looking at the initial hysteresis loops can (usually) tell

Nca = Uwxa_full.shape[0]

for line in range(10):
    U = Uwxa_full[line, :-3]*10**Uwxa_full[line, -1]
    w = Uwxa_full[line, -3]
    times, disp, forces_iwan = nlutils.hysteresis_loop(1 << 7, h, np.atleast_2d(U).T, w, 'iwan', [kt, Fs, chi, beta])
    times, disp, forces_arc = nlutils.hysteresis_loop(1 << 7, h, np.atleast_2d(U).T, w, 'arctangent', [Kt_true, b, s])
    plt.plot(disp, forces_iwan)
    plt.plot(disp, forces_arc)
    plt.title(f"line, arc_ks: {arctangent_force.secant_stiffness(np.max(U))}")
    plt.show()

#%% 




line = 10 #arbitrary
X0 = np.atleast_2d(Uwxa_full[line, :-3]).T * 10**Uwxa_full[line, -1]

Ut = hutils.time_series_deriv(1 << 11, h, X0, 0)
Utdot = hutils.time_series_deriv(1 << 11, h, X0, 1)

# Compute forces
nonhysteretic_force = arctangent_force.secant_stiffness(2 * Ut.max()) * Ut
arctangent_force_history = arctangent_force.local_force_history(Ut, Utdot, h, cst, X0[0])
iwan_force_history = test_iwan.local_force_history(Ut, Utdot, h, cst, X0[0])

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Arctangent Hysteresis Loop
axes[0].plot(Ut, arctangent_force_history[0], '*', label="Arctangent Hysteresis")
axes[0].plot(Ut, nonhysteretic_force, label="Nonhysteretic Force")
axes[0].set_title("Hysteresis Loop Arctangent")
axes[0].legend()

# Plot Iwan Hysteresis Loop
axes[1].plot(Ut, iwan_force_history[0], label="Iwan Hysteresis")
axes[1].set_title(f"Hysteresis Loop Iwan: ks = {max(iwan_force_history[0])/max(Ut)}")
axes[1].legend()

# Show the figure
plt.tight_layout()
plt.show()

def iwan_avg_stiffness(u, kt, Fs, beta, chi):
    x = (kt * (beta + 1) * u) / (Fs * (beta + chi + 2))
    I = betainc(beta + 2, chi + 1, x)  # Note: beta + 2 (not beta + 1)
    return kt * (1 - (Fs * (beta + chi + 2) / (kt * (beta + 1) * u) * I))

iwan_kt = iwan_avg_stiffness(beginning_amplitude, kt, Fs, beta, chi)
print(f"Deepseek Guess for iwan Kt: {iwan_kt}")

#arctangent_force.damping_error(Q, T, 10**Uwxa_full[-1, -1], Uwxa_full[-1, -2], Uwxa_full[-1, -3], kt, b, s, m, c, verbose=True)

ks_iwan = np.zeros((Uwxa_full.shape[0]))
ks_arc = np.zeros((Uwxa_full.shape[0]))
X0norm = np.zeros((Uwxa_full.shape[0]))

for line in range(Uwxa_full.shape[0]):
    print(line)
    Uwxa = Uwxa_full[line, :]
    omega = Uwxa[-3]
    amplitude = Uwxa[-1]
    X0 = np.atleast_2d(Uwxa_full[line, :-3]).T*(10**Uwxa_full[line, -1])
    X0norm[line] = np.linalg.norm(X0)
    Ut = hutils.time_series_deriv(1 << 7, h, X0, 0)
    Utdot = hutils.time_series_deriv(1 << 7, h, X0, 1)
   
    arc_force = arctangent_force.local_force_history(Ut, Utdot, h, cst, X0[0])[0]
    iwan_force = test_iwan.local_force_history(Ut, Utdot, h, cst, X0[0])[0]
    ks_iwan[line] = (iwan_force[np.argmax(Ut)] - iwan_force[np.argmin(Ut)])/(max(Ut) - min(Ut))
    ks_arc[line] = (arc_force[np.argmax(Ut)] - arc_force[np.argmin(Ut)])/(max(Ut) - min(Ut))

    
plt.plot(Uwxa_full[:, -1], ks_iwan)
plt.plot(Uwxa_full[:, -1], ks_arc)
plt.legend(('Iwan secant stiffness', 'Arctangent secant stiffness'))
plt.show()

plt.plot(Uwxa_full[:, -1], np.log10(X0norm))
plt.show()


#%% Display secant stiffness wrt natural frequency


ks = [arctangent_force.secant_stiffness(10**Uwxa_full[line, -1]) for line in range(Uwxa_full.shape[0])]

plt.plot(Uwxa_full[:, -3], ks)
plt.title("Arctangent Model: omega, ks")
plt.show()

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.title("Iwan: A, omega")
plt.show()

plt.plot(Uwxa_full[:, -1], ks)
plt.title("Arctangent Model: A, ks")
plt.show()

#%% 

###############################################################################
####### 1. Load System Matrices                                         #######
###############################################################################
system_fname = './data/brb_iwan4_mesh.mat'
system_matrices = sio.loadmat(system_fname)

M = np.array([[m]])
c = 0.005
K = np.array([[k]])
Ndof = M.shape[1]

Q = np.array([[1]])
T = np.array([[1]])

Nnl, Nnodes = Q.shape


###############################################################################
####### 2. Establish Vibration System                                   #######
###############################################################################

damp_ab = [c, 0.0]
# Proportional damping arbitrary value

vib_sys = VibrationSystem(M, K, ab=damp_ab)

Q = np.array([[1]])
T = np.array([[1]])

nlforce = ArctangentStiffness(Q, T, kt, b, s)


vib_sys.add_nl_force(nlforce)

Astart = -10
Aend = 0

# Normal - settings for higher accuracy as used in previous papers
h_max = 3  # harmonics 0, 1, 2, 3
Nt = 1 << 7  # 2**7 = 128 AFT steps

ds = 0.008
dsmax = 0.015
dsmin = 0.002
# Adjust weighting of amplitude v. other in continuation to hopefully
# reduce turning around. Higher puts more emphasis on continuation
# parameter (amplitude)
FracLam = 1

static_solver = NonlinearSolver()

eigvals, eigvecs = static_solver.eigs(K, M)




###############################################################################
####### 13. EPMC Initial Guess                                          #######
###############################################################################

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Fl = np.zeros(Nhc*Ndof)

# Forcing Vector
Fl[1] = 1  # Cosine force vector

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Mode Shape (from linearized system for prediction)
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = c  # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta


Uwxa0[-1] = Astart


'''
For some reason why, the continuation guess method here is very bad. For now, 
the guess is the first step from Iwan
'''

Uwxa0 = Uwxa_full[-1, :]


###############################################################################
####### 15. EPMC Continuation                                           #######
###############################################################################

# This block actually executes the full continuation for the EPMC solution.


def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 1000,  # balance with reform_freq
               'reform_freq': 2,  # >1 corresponds to BFGS
               'verbose': True,
               'xtol': None,  # Just use the one passed from continuation
               'rtol': 1e-9,
               'etol': None,
               'xtol_rel': 1e-6,
               'rtol_rel': None,
               'etol_rel': None,
               'stopping_tol': ['xtol'],  # stop on xtol
               # accept solution on these
               'accepting_tol': ['xtol_rel', 'rtol']
               }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolver()

continue_config = {'DynamicCtoP': True,
                   'TargetNfev': 4,
                   'MaxSteps': 5000,  # May need more depending on ds and dsmin
                   'dsmin': dsmin,
                   'dsmax': dsmax,
                   'verbose': 1,
                   'xtol': 1e-6*np.sqrt(Uwxa0.shape[0]),
                   'corrector': 'Ortho',  # Ortho, Pseudo
                   'nsolve_verbose': True,
                   'FracLam': FracLam,
                   'FracLamList': [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop': 0.05  # stop if backtracks to before lam0
                   }


print(f"EPMC INITIAL RESIDUAL: {epmc_fun(Uwxa0, False)}")


# The conditioning of the static displacements should be small since these
# displacements are small, but very important
CtoPstatic = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-5)

# Increasing delta means that these cofficients will be smaller in conditioned
# space. - reduces importance of higher harmonics when calculating arc length
CtoP = hutils.harmonic_wise_conditioning(Uwxa0, Ndof, h, delta=1e-3)

# Allow different CtoP for static displacements than harmonics.
CtoP[:Ndof] = CtoPstatic[:Ndof]

# Exactly take damping and frequency regardless of delta for conditioning
CtoP[-3:-1] = np.abs(Uwxa0[-3:-1])

# scale so step size is similar order as fraction of total distance from
# start to end
CtoP[-1] = np.abs(Aend-Astart)

R0, dRdX0, dRda = epmc_fun(Uwxa0)
print("Norm of initial solution residual: {}".format(np.linalg.norm(R0)))

cont_solver = Continuation(epmc_solver, ds0=ds, CtoP=CtoP,
                           config=continue_config)

t0 = time.time()

Uwxa_full_arctangent = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))

freqs = Uwxa_full_arctangent[:, -3]
freq_diffs = freqs - freqs[0]
amps = Uwxa_full_arctangent[:, -1]

plt.plot(amps, freqs)
plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Natural Frequency")
plt.legend(("Arctangent Stiffness", "Iwan"))
plt.show()


for dof in range(Uwxa_full[-1, :].shape[0] - 1):
    plt.plot(Uwxa_full[:, -1], Uwxa_full[:, dof])
    plt.plot(Uwxa_full_arctangent[:, -1], Uwxa_full_arctangent[:, dof])
    if dof > 0 and dof%2 == 1:
        plt.title(f"Harmony: {(dof + 1)//2}c")
    elif dof >0:
        plt.title(f"Harmony: {(dof + 1)//2}s")
    else:
        plt.title("Harmony 0")
    plt.legend(("Iwan", "Arctangent Stiffness"))
    plt.show()
    
    
    
#%% Printing Hysteresis Loops

    