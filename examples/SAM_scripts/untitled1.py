'''
Nonlinear Model test with Multiple DOFs in Sample System
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tmdsimpy.nlforces.sigmoid_stiffness import SigmoidStiffness
from tmdsimpy.nlforces.iwan4_element import *
from tmdsimpy.nlforces.bouc_wen import *
from tmdsimpy.utils.harmonic import *
import matplotlib.pyplot as plt
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.vibration_system import VibrationSystem
from sdof_iwan_epmc import sdof_uwxa_full
from scipy.special import betainc
import tmdsimpy.nlutils as nlutils
from scipy import io as sio
import time




M = np.diag([10,10, 10, 100, 0.001])
C = M*0.005
c = 0.005
K = 1e10*np.array([
    [2,  -1, 0, 0, 0],
    [ -1, 3, -2, 0, 0],
    [0, -2, 3, -1, 0],
    [0, 0, -1, 1+0.001, -0.001],
    [0, 0, 0, -0.001, 0.001]
])


Ndof = M.shape[0]

Q = np.array([[1 , 0, 0, 0, 0],
              [0, 0, -1, 1, 0]])
T = Q.T

Nnl = Q.shape[0]
'''
BRB Iwan Parameters for some testing
'''
lparams = [5.530289365884571, 15.859238733093676, -0.999870602783605, -5.746331062210725, 16.888652840667923] #Best Freq

lparams[2] = -0.023
lpsci = [1, 1, 0, 1, 1]

iwan_parameters = [10 ** lparams[i] if lpsci[i] == 1 else lparams[i] for i in range(len(lparams))]
kt_brb = iwan_parameters[1] * 0.000385483218541205
Fs_brb = iwan_parameters[0] * 0.000385483218541205


kt = 1e7
Fs = 0.2 *1e6 # N, Match Jenkins
chi = 0.0  # Have a more full hysteresis loop than chi=0.0
beta = 0.0  # Smooth Transition

iwan_force = Iwan4Force(np.atleast_2d(Q[0, :]), np.atleast_2d(T[:, 0]).T, kt, Fs, chi, beta)
iwan_force2= Iwan4Force(np.atleast_2d(Q[1, :]), np.atleast_2d(T[:, 1]).T, kt, Fs, chi, beta)

vib_sys = VibrationSystem(M, K, C = C)
vib_sys.add_nl_force(iwan_force)
vib_sys.add_nl_force(iwan_force2)

Astart = -8
Aend = 3

# Normal - settings for higher accuracy as used in previous papers
h_max = 3  # harmonics 0, 1, 2, 3
Nt = 1 << 7  # 2**7 = 128 AFT steps
h = np.array(range(h_max+1))
Nhc = hutils.Nhc(h)
mode_ind = 0
ds = 0.008
dsmax = 0.015
dsmin = 0.002
# Adjust weighting of amplitude v. other in continuation to hopefully
# reduce turning around. Higher puts more emphasis on continuation
# parameter (amplitude)
FracLam = 1

###
# 1. Static Analysis for Iwan Model
###

static_solver = NonlinearSolver()
Fv = np.zeros(Ndof)
# Forcing Vector
Fv[-1] = 1  # Cosine force vector at arbitrary dof

_, eigvecs = static_solver.eigs(K, M)
X0 = eigvecs[:, mode_ind]

static_fun = lambda U, calc_grad = True: vib_sys.static_res(U, Fv)
R0, dR0dX = static_fun(X0)

static_config={'max_steps' : 30,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
static_solver = NonlinearSolver() 


Xpre, R, dRdX, sol = static_solver.nsolve(static_fun, X0,
                                          verbose=True, xtol=1e-13)

vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

Kpre_iwan = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre_iwan, M)

###
# 2. Build EPMC Guess for Iwan Model
###


Fl = np.zeros(Nhc*Ndof)
Fl[:Ndof] = Fv
Fl[2*Ndof] = 1 #Cosine Forcing


Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Mode Shape (from linearized system for prediction)

Uwxa0[Ndof:2*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = c  # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta


Uwxa0[-1] = Astart

###
# 3. Run EPMC
###

def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 100,  # balance with reform_freq
               'reform_freq': 2,  # >1 corresponds to BFGS
               'verbose': True,
               'xtol': None,  # Just use the one passed from continuation
               'rtol': 1e-9,
               'etol': None,
               'xtol_rel': 1e0,
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
                   'xtol': 1e-4*np.sqrt(Uwxa0.shape[0]),
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

Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))


###
# 4. Plot results for Iwan
###

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.xlabel("Log Amplitude")
plt.ylabel("Frequency")
plt.title("Backbone Curve Frequency: Iwan Model")
plt.show()

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -2])
plt.xlabel("Log Amplitude")
plt.ylabel("Self-Excitation")
plt.title("Backbone Curve Self-Excitation: Iwan Model")
plt.show()

#%% Arctangent Model

Uwxa_iwan_nl = nlutils.transform_to_nonlinear(Uwxa_full, Q, Ndof, Nnl)
Nc_iwan = Uwxa_full.shape[0]

Unorm0 = np.zeros(Nc_iwan)
Unorm1 = np.zeros(Nc_iwan)

#For one nonlinear degree
for line in range(Nc_iwan):
    Unorm0[line] = np.linalg.norm(Uwxa_iwan_nl[line, 0:-3:Nnl]*10**Uwxa_iwan_nl[line, -1])
    Unorm1[line] = np.linalg.norm(Uwxa_iwan_nl[line, 1:-3:Nnl]*10**Uwxa_iwan_nl[line, -1])



b0, s0 = SigmoidStiffness.incomplete_slip_parameters(np.log10(Unorm0), Uwxa_full[:, -3], None, verbose=True)
b1, s1 = SigmoidStiffness.incomplete_slip_parameters(np.log10(Unorm1), Uwxa_full[:, -3], None, verbose=True)

b0, b1 = b0, b1

phi_max = Fs*(1 + beta)/(kt * (beta + (chi + 1)/(chi + 2)))
Kt0_true = kt*(1 - (Unorm0[0]/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))
Kt1_true = kt*(1 - (Unorm1[0]/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))


U0_iwan = Uwxa_iwan_nl[0, 0:-3:Nnl]*10**Uwxa_iwan_nl[0, -1]
U1_iwan = Uwxa_iwan_nl[0, 1:-3:Nnl]*10**Uwxa_iwan_nl[0, -1]

beginning_displacement0 = np.linalg.norm(U0_iwan)
beginning_displacement1 = np.linalg.norm(U1_iwan)

w0_iwan = Uwxa_iwan_nl[0, -3]
times, disp, forces_iwan0 = nlutils.hysteresis_loop(1 << 10, h, np.atleast_2d(U0_iwan).T, w0_iwan, 'iwan', [kt, Fs, chi, beta])
ks_iwan_initial = (forces_iwan0[np.argmax(disp)] - forces_iwan0[np.argmin(disp)])/(max(disp) - min(disp))

times, disp, forces_iwan0 = nlutils.hysteresis_loop(1 << 10, h, np.atleast_2d(U1_iwan).T, w0_iwan, 'iwan', [kt, Fs, chi, beta])
ks_iwan_initial = (forces_iwan0[np.argmax(disp)] - forces_iwan0[np.argmin(disp)])/(max(disp) - min(disp))
print(f"Initial Iwan Tangent Stiffness: {ks_iwan_initial}")

sig0 = SigmoidStiffness(np.atleast_2d(Q[0, :]), np.atleast_2d(T[:, 0]).T, kt, b0, s0)
sig1 = SigmoidStiffness(np.atleast_2d(Q[1, :]), np.atleast_2d(T[:, 1]).T, kt, b1, s1)

vib_sys_arc = VibrationSystem(M, K, C = C)
vib_sys_arc.add_nl_force(sig0)
vib_sys_arc.add_nl_force(sig1)

Astart = -8
Aend = 3

# Normal - settings for higher accuracy as used in previous papers
h_max = 3  # harmonics 0, 1, 2, 3
Nt = 1 << 7  # 2**7 = 128 AFT steps

ds = 0.02
dsmax = 0.15*4
dsmin = 0.002
# Adjust weighting of amplitude v. other in continuation to hopefully
# reduce turning around. Higher puts more emphasis on continuation
# parameter (amplitude)
FracLam = 1.0


###
#   1. Static Analysis for Arctangent Model
###

static_solver = NonlinearSolver()
Fv = np.zeros(Ndof)
# Forcing Vector
Fv[-1] = 1  # Cosine force vector at arbitrary dof

_, eigvals = static_solver.eigs(K, M)
X0 = eigvals[:, mode_ind]

static_fun = lambda U, calc_grad = True: vib_sys_arc.static_res(U, Fv)
R0, dR0dX = static_fun(X0)

static_config={'max_steps' : 300,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

Xpre, R, dRdX, sol = static_solver.nsolve(static_fun, X0,
                                          verbose=True, xtol=1e-13)

#vib_sys_arc.update_force_history(Xpre)
vib_sys_arc.reset_real_mu()

Rpre, dRpredX = vib_sys_arc.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre, M)

###
# 2. Build EPMC Guess for Arctangent Model
###


Fl = np.zeros(Nhc*Ndof)
Fl[:Ndof] = Fv
Fl[2*Ndof] = 1 #Cosine Forcing


Uwxa0_arc = np.zeros(Nhc*Ndof + 3)

# Mode Shape (from linearized system for prediction)

Uwxa0_arc[Ndof:2*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0_arc[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = c  # This is what mass/stiff prop damping should give
Uwxa0_arc[-2] = 2*Uwxa0[-3]*zeta


Uwxa0_arc[-1] = Astart


###
# 3. Run EPMC
###

def epmc_fun(Uwxa, calc_grad=True): return vib_sys_arc.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 10000,  # balance with reform_freq
               'reform_freq': 2,  # >1 corresponds to BFGS
               'verbose': True,
               'xtol': None,  # Just use the one passed from continuation
               'rtol': 1e-9,
               'etol': None,
               'xtol_rel': 1e-1*np.sqrt(Uwxa0.shape[0]),
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
                   'MaxSteps': 10000,  # May need more depending on ds and dsmin
                   'dsmin': dsmin,
                   'dsmax': dsmax,
                   'verbose': 1,
                   'xtol': 1e-7*np.sqrt(Uwxa0.shape[0]),
                   'corrector': 'Ortho',  # Ortho, Pseudo
                   'nsolve_verbose': True,
                   'FracLam': FracLam,
                   'FracLamList': [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop': 0.05  # stop if backtracks to before lam0
                   }



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

###
# 4. Plot results for Arctangent
###

plt.plot(Uwxa_full_arctangent[:, -1], Uwxa_full_arctangent[:, -3])
plt.xlabel("Log Amplitude")
plt.ylabel("Frequency")
plt.title("Backbone Curve Frequency: Arctangent Model")
plt.show()

plt.plot(Uwxa_full_arctangent[:, -1], Uwxa_full_arctangent[:, -2])
plt.xlabel("Log Amplitude")
plt.ylabel("Self-Excitation")
plt.title("Backbone Curve Self-Excitation: Arctangent Model")
plt.show()

plt.plot(Uwxa_full_arctangent[:, -1], Uwxa_full_arctangent[:, -3])
plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.xlabel("Log Amplitude")
plt.ylabel("Frequency")
plt.title("Backbone Curve Frequency: Arctangent Model")
plt.show()


Nhc = Uwxa_full_arctangent.shape[0]


Uwxa_full_nl_arc = nlutils.transform_to_nonlinear(Uwxa_full_arctangent, Q, Ndof, Nnl)

'''
for line in range(Nhc):
    if line in range(1, 230):
        times, disp, forces = nlutils.hysteresis_loop(1 << 9, h, np.atleast_2d(Uwxa_iwan_nl[line, 0:-3:Nnl]*10**Uwxa_iwan_nl[line, -1]).T, Uwxa_iwan_nl[line, -3], 'arctangent', [Kt0_true, b0, s0])
        plt.plot(disp, forces)
        plt.show()
'''
   