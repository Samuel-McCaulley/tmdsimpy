import sys
import os
from scipy import io as sio
import numpy as np
import time
sys.path.append('../..')
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.nlforces.bouc_wen import BoucWenForce
from tmdsimpy.nlforces.iwan4_element import Iwan4Force
from tmdsimpy.nlforces.general_poly_stiffness import GenPolyForce
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.jax.solvers import NonlinearSolverOMP
import tmdsimpy.nlutils as hutils_sam

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.continuation import Continuation

import matplotlib.pyplot as plt

solve_data = np.load('data/iwan_epmc.npz')


M = np.array([[1]])
K = np.array([[1e6]])
Q = np.array([[1]])
T = np.array([[1]])
damp_ab = [0.087e-2*2*(168.622*2*np.pi), 0.0]
vib_sys = VibrationSystem(M, K, ab=damp_ab)

iwan_parameters = solve_data['iwan_parameters']

iwan_force = Iwan4Force(Q, T, iwan_parameters[0], 
                        iwan_parameters[1], iwan_parameters[2], 
                        iwan_parameters[3])

Astart = -9
Aend = -4.7

# Normal - settings for higher accuracy as used in previous papers
h_max = 3 # harmonics 0, 1, 2, 3
Nt = 1<<7 # 2**7 = 128 AFT steps 

ds = 0.08
dsmax = 0.125*1.4
dsmin = 0.02

###############################################################################
####### 3. Prestress Analysis                                           #######
###############################################################################
h_max = 3
h = np.array(range(h_max + 1))
Nhc = hutils.Nhc(h)
X0 = np.array([[1e-6]]) #Figure out what this is

Fv = 1
prestress = 12249.0 #stolen from brb_epmc






pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)
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

t0 = time.time()


Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

t1 = time.time()

print('Residual norm: {:.4e}'.format(np.linalg.norm(R)))

print('Static Solution Run Time : {:.3e} s'.format(t1 - t0))

vib_sys.update_force_history(Xpre)
vib_sys.reset_real_mu()

###############################################################################
####### 11. Updated Eigenvalue Analysis After Prestress                 #######
###############################################################################

Rpre, dRpredX = vib_sys.static_res(Xpre, Fv*prestress)

sym_check = np.max(np.abs(dRpredX - dRpredX.T))
print('Symmetrix matrix has a maximum error/max value of: {}'.format(
                                         sym_check / np.abs(dRpredX).max()))

print('Using using  (Kpre + Kpre.T)/2 version for eigen analysis')

Kpre = (dRpredX + dRpredX.T) / 2.0 #Gets a really off-kilter prestress for some reason

#Imported prestress stiffness matrix

K_stat_imported = sio.loadmat('./data/K_static.mat')['dRstat']


eigvals, eigvecs = static_solver.eigs(Kpre, M, 
                                      subset_by_index=[0, 9])

eigvals, eigvecs = hutils_sam.process_eigenpairs(eigvals, eigvecs)
print(f"First Eiegenvalue: {eigvals[0]}")
#Fix negative eigenvalues -- ask Brake about these

###############################################################################
####### 13. EPMC Initial Guess                                          #######
###############################################################################

h = np.array(range(h_max+1))

Nhc = hutils.Nhc(h)

Ndof = vib_sys.M.shape[0]


Fl = np.zeros(Nhc*Ndof)

# Static Forces
Fl[:Ndof] = prestress*Fv # EPMC static force

# EPMC phase constraint - No cosine component at accel
# Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] 
# We don't have an R component, so I'm not sure how to do this hehehe


Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements (prediction for 0th harmonic)
Uwxa0[:Ndof] = Xpre

# Mode Shape (from linearized system for prediction)
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = 0.087e-2*2*(168.622*2*np.pi)*M/K # This is what mass/stiff prop damping should give
Uwxa0[-2] = 2*Uwxa0[-3]*zeta


Uwxa0[-1] = Astart

###############################################################################
####### 15. EPMC Continuation                                           #######
###############################################################################

# This block actually executes the full continuation for the EPMC solution.

epmc_fun = lambda Uwxa, calc_grad=True : vib_sys.epmc_res(Uwxa, Fl, h, Nt=Nt, 
                                                          calc_grad=calc_grad)
epmc_config={'max_steps' : 12, # balance with reform_freq
            'reform_freq' : 2, #>1 corresponds to BFGS 
            'verbose' : True, 
            'xtol'    : None, # Just use the one passed from continuation
            'rtol'    : 1e-9,
            'etol'    : None,
            'xtol_rel' : 1e-6, 
            'rtol_rel' : None,
            'etol_rel' : None,
            'stopping_tol' : ['xtol'], # stop on xtol 
            'accepting_tol' : ['xtol_rel', 'rtol'] # accept solution on these
            }

# Custom Newton-Raphson solver
epmc_solver = NonlinearSolver()

FracLam = 0.5

continue_config = {'DynamicCtoP': True, 
                   'TargetNfev' : 4,
                   'MaxSteps'   : 250, # May need more depending on ds and dsmin
                   'dsmin'      : dsmin,
                   'dsmax'      : dsmax,
                   'verbose'    : 1,
                   'xtol'       : 1e-7*np.sqrt(Uwxa0.shape[0]), 
                   'corrector'  : 'Ortho', # Ortho, Pseudo
                   'nsolve_verbose' : True,
                   'FracLam' : FracLam,
                   'FracLamList' : [0.9, 0.1, 1.0, 0.0],
                   'backtrackStop' : 0.05 # stop if backtracks to before lam0
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

Uwxa_full = cont_solver.continuation(epmc_fun, Uwxa0, Astart, Aend)

t1 = time.time()

print('Continuation solve time: {: 8.3f} seconds'.format(t1-t0))


freqs = Uwxa_full[:, -3]/2/np.pi
amps = Uwxa_full[:, -1]

plt.plot(amps, freqs)
plt.xlabel("Log Modal Amplitude")
plt.ylabel("Natural Frequency")
plt.title("SDOF Iwan: " + np.array2string(np.round(iwan_parameters, 5)))
plt.show()
