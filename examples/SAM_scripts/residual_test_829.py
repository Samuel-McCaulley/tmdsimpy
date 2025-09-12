'''
Nonlinear Model test with Multiple DOFs in Sample System
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tmdsimpy.nlforces.arcstiffness import ArcStiffness
from tmdsimpy.nlforces.vector_iwan4 import *
from tmdsimpy.utils.harmonic import *
import matplotlib.pyplot as plt
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.jax.solvers import NonlinearSolverOMP
from tmdsimpy.solvers import NonlinearSolver

from tmdsimpy.vibration_system import VibrationSystem
from sdof_iwan_epmc import sdof_uwxa_full
import tmdsimpy.nlutils as nlutils
from scipy import io as sio
import time
#%% Iwan Modeling

M = np.diag([1])
C = M*0.005
c = 0.005

K = 1 * np.array([
    [1]
])


Ndof = M.shape[0]


Q = np.array([[1]])
T = Q.T

kt = 9999
Fs = 100  # N, Match Jenkins
chi = 0.0  # Have a more full hysteresis loop than chi=0.0
beta = 0.0  # Smooth Transition

iwan_force = VectorIwan4(np.atleast_2d(Q[0, :]), np.atleast_2d(T[:, 0]).T, kt, Fs, chi, beta)

vib_sys = VibrationSystem(M, K, C = C)
vib_sys.add_nl_force(iwan_force)
#vib_sys.add_nl_force(iwan_force1)
ref_nlforces = vib_sys.nonlinear_forces

Astart = -10
Aend = 3

# Normal - settings for higher accuracy as used in previous papers
h_max = 1  # harmonics 0, 1, 2, 3
Nt = 1 << 9  # 2**7 = 128 AFT steps
h = np.array(range(h_max+1))
Nhc = hutils.Nhc(h)
mode_ind = 0

static_solver = NonlinearSolver()
Fv = np.zeros(Ndof)
# Forcing Vector
Fv[-1] = 1 * 10 ** Astart  # Cosine force vector at arbitrary dof

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
#static_solver = NonlinearSolverOMP(config=static_config)


Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

#vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre_iwan = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

eigvals, eigvecs = static_solver.eigs(Kpre_iwan, M) #Temporary
Fl = np.zeros(Nhc*Ndof)
Fl[:Ndof] = Fv
Fl[2*Ndof] = 1 #Sine Forcing


Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Mode Shape (from linearized system for prediction)

Uwxa0[Ndof:2*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = c  # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta
#Uwxa0[-2] = 0 #This is the modification to damping that is not originally correct -- I think

Uwxa0[-1] = Astart




def epmc_fun(Uwxa, calc_grad=True): return vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)


Uwxa = np.array([1e-10, 1, 0.2, 100, 0.005, -10])


R, dRdUwx, dRda = epmc_fun(Uwxa)


B = 94.62928669
S = 0.00837318
K_T = 9999


arc_force = ArcStiffness(Q, T, K_T, B, S)

vib_sys_arc = VibrationSystem(M, K, C = C)
vib_sys_arc.add_nl_force(arc_force)

def epmc_fun_arc(Uwxa, calc_grad=True): return vib_sys_arc.epmc_res(Uwxa, Fl, h, Nt=Nt,
                                                            calc_grad=calc_grad)

R_arc, dRdUwx_arc, dRda_arc = epmc_fun_arc(Uwxa)


deltaX_iwan = np.linalg.solve(dRdUwx, R)
deltaX_arc =  np.linalg.solve(dRdUwx_arc, R_arc)


