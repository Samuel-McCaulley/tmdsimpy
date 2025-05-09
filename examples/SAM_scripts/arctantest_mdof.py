'''
Nonlinear Model test with Multiple DOFs in Sample System
'''
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
from tmdsimpy.jax.solvers import NonlinearSolverOMP
from tmdsimpy.vibration_system import VibrationSystem
from sdof_iwan_epmc import sdof_uwxa_full
import tmdsimpy.nlutils as nlutils
from scipy import io as sio
import time


M = np.diag([1, 0.5, 0.25, 0.125])
C = M*0.005
c = 0.005
K = 1*np.array([
    [1+2,  -2,    0,    0],  # k₁ + k₂ = 3
    [ -2, 2+3,   -3,    0],  # k₂ + k₃ = 5
    [  0,   -3, 3+4,   -4],  # k₃ + k₄ = 7
    [  0,    0,   -4,   4]   # k₄ = 4
])


Ndof = M.shape[0]

Q = np.array([[1 , 0, 0, 0],
              [0, 0, 1, -1]])
T = Q.T

kt = 1.25*1
Fs = 0.2*10  # N, Match Jenkins
chi = -0.99  # Have a more full hysteresis loop than chi=0.0
beta = 0.0  # Smooth Transition

iwan_force = Iwan4Force(np.atleast_2d(Q[0, :]), np.atleast_2d(T[:, 0]).T, kt, Fs, chi, beta)
iwan_force_2 = Iwan4Force(np.atleast_2d(Q[1, :]), np.atleast_2d(T[:, 1]).T, kt, Fs, chi, beta)


vib_sys = VibrationSystem(M, K, C = C)
vib_sys.add_nl_force(iwan_force)
vib_sys.add_nl_force(iwan_force_2)

Astart = -8
Aend = 3

# Normal - settings for higher accuracy as used in previous papers
h_max = 3  # harmonics 0, 1, 2, 3
Nt = 1 << 7  # 2**7 = 128 AFT steps
h = np.array(range(h_max+1))
Nhc = hutils.Nhc(h)
mode_ind = 0
ds = 0.0008
dsmax = 0.0015
dsmin = 0.0002
# Adjust weighting of amplitude v. other in continuation to hopefully
# reduce turning around. Higher puts more emphasis on continuation
# parameter (amplitude)
FracLam = 1

static_solver = NonlinearSolver()
Fv = np.zeros(Ndof)
# Forcing Vector
Fv[-1] = 1  # Cosine force vector at arbitrary dof

eigvals_pre, eigvecs_pre = static_solver.eigs(K, M)

X0 = np.real(eigvecs_pre[:, mode_ind])

pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv)
R0, dR0dX = pre_fun(X0)
print('Residual norm of initial guess: {:.4e}'.format(np.linalg.norm(dR0dX)))

static_config={'max_steps' : 30,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
static_solver = NonlinearSolver() 


Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

#vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre, M) #Temporary

#%%
# This block actually executes the full continuation for the EPMC solution.


Fl = np.zeros(Nhc*Ndof)
Fl[:Ndof] = Fv
Fl[2*Ndof] = 1 #Cosine Forcing


Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Mode Shape (from linearized system for prediction)

Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = c  # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta


Uwxa0[-1] = Astart


def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 300,  # balance with reform_freq
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
epmc_solver = NonlinearSolverOMP(config=epmc_config)

continue_config = {'DynamicCtoP': True,
                   'TargetNfev': 4,
                   'MaxSteps': 5000,  # May need more depending on ds and dsmin
                   'dsmin': dsmin,
                   'dsmax': dsmax,
                   'verbose': 1,
                   'xtol': 1e-2*np.sqrt(Uwxa0.shape[0]),
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


Uwxa_full = Uwxa_full[6:, :] #Clip the bad parts

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.show()

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -2])
plt.show()



#%% Arctangent

Uwxa_nl_full = nlutils.transform_to_nonlinear(Uwxa_full, Q, Ndof, Q.shape[0])

U0norm = np.linalg.norm(Uwxa_nl_full[:, 0:-3:2], axis = 1)*10**Uwxa_nl_full[:, -1]
U1norm = np.linalg.norm(Uwxa_nl_full[:, 1:-3:2], axis = 1)*10**Uwxa_nl_full[:, -1]

b0, s0 = ArctangentStiffness.incomplete_slip_parameters(np.log10(U0norm), Uwxa_nl_full[:, -3], Uwxa_nl_full[:, -2], verbose=True)
b1, s1 = ArctangentStiffness.incomplete_slip_parameters(np.log10(U1norm), Uwxa_nl_full[:, -3], Uwxa_nl_full[:, -2], verbose=True)


phi_max = Fs*(1 + beta)/(kt * (beta + (chi + 1)/(chi + 2)))


Kt0_true = kt*(1 - (U0norm[0]/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))
Kt1_true = kt*(1 - (U1norm[0]/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))

arctan0 = ArctangentStiffness(np.atleast_2d(Q[0, :]), np.atleast_2d(T[:, 0]).T, Kt0_true, b0, s0)
arctan1 = ArctangentStiffness(np.atleast_2d(Q[1, :]), np.atleast_2d(T[:, 1]).T, Kt1_true, b1, s1)

vib_sys = VibrationSystem(M, K, C = C)
vib_sys.add_nl_force(arctan0)
vib_sys.add_nl_force(arctan1)

Astart = -8
Aend = 2

# Normal - settings for higher accuracy as used in previous papers
h_max = 3  # harmonics 0, 1, 2, 3
Nt = 1 << 7  # 2**7 = 128 AFT steps

ds = 0.0008
dsmax = 0.0015
dsmin = 0.0002
# Adjust weighting of amplitude v. other in continuation to hopefully
# reduce turning around. Higher puts more emphasis on continuation
# parameter (amplitude)
FracLam = 1.0

static_solver = NonlinearSolver()

Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre, M)

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Fl = np.zeros(Nhc*Ndof)

# Forcing Vector
Fl[Ndof] = 1  # Cosine force vector

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

Uwxa0 = Uwxa_full[0, :]

# This block actually executes the full continuation for the EPMC solution.


def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


epmc_config = {'max_steps': 5000,  # balance with reform_freq
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
                   'xtol': 1e-3*np.sqrt(Uwxa0.shape[0]),
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

Uwxa_full_arctan = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))


plt.plot(Uwxa_full_arctan[:, -1], Uwxa_full_arctan[:, -3])
plt.show()

plt.plot(Uwxa_full_arctan[:, -1], Uwxa_full_arctan[:, -2])
plt.show()

print(b0, s0, b1, s1)

plt.plot()

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -3])
plt.plot(Uwxa_full_arctan[:, -1], Uwxa_full_arctan[:, -3])
plt.show()

plt.plot(Uwxa_full[:, -1], Uwxa_full[:, -2])
plt.plot(Uwxa_full_arctan[:, -1], Uwxa_full_arctan[:, -2])

plt.show()

