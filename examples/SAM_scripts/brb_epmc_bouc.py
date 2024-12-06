import sys
import os
from scipy import io as sio
import numpy as np
import time
sys.path.append('../..')
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.nlforces.bouc_wen import BoucWenForce
from tmdsimpy.nlforces.general_poly_stiffness import GenPolyForce
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.jax.solvers import NonlinearSolverOMP
import tmdsimpy.nlutils as hutils_sam

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.continuation import Continuation

import matplotlib.pyplot as plt



###############################################################################
####### 1. Load System Matrices                                         #######
###############################################################################
system_fname = './data/brb_iwan4_mesh.mat'
system_matrices = sio.loadmat(system_fname)

M = system_matrices['M']
K = system_matrices['K']
Ndof = M.shape[1]

Q = np.array(system_matrices['Qxyn'])
T = np.array(system_matrices['Txyn'])

Nnl,Nnodes = Q.shape


###############################################################################
####### 2. Establish Vibration System                                   #######
###############################################################################

damp_ab = [0.087e-2*2*(168.622*2*np.pi), 0.0]
#Proportional damping arbitrary value

vib_sys = VibrationSystem(M, K, ab=damp_ab)


#lparams = [12.469255402222181, 8.33731378, 4.737470329999999, 0.30489182,16.888652840667923]
lparams = [12.790433482222179, 6.51131523, 7.539651399999999, 1.02588743, 16.888652840667923]

lpsci = [1, 1, 1, 0, 1]

bouc_parameters = [10 ** lparams[i] if lpsci[i] == 1 else lparams[i] for i in range(len(lparams))]

patch_areas = [0.000385483218541205, 0.000715750926938309, 0.000715750512690334, 0.000715750596958764, 0.000385483218541205] #Iwan parameters

X0 = np.squeeze(system_matrices['X0'])

u0 = Q @ X0

uxyn_0 = np.column_stack((Q[0::3][:] @ X0, Q[1::3][:] @ X0, Q[2::3][:] @ X0))

for i in range(Nnl):
    Ls = Q[i:i+1, :]
    Lf = T[:, i:i+1]
    
    A = bouc_parameters[0] * patch_areas[i // 3]
    beta = bouc_parameters[1]
    gamma = bouc_parameters[2]
    n = bouc_parameters[3]
    Kn = bouc_parameters[4] * patch_areas[i // 3]
    
    if i == 9:
        print(f"Alog: {np.log10(A)}")
        print(f"betalog: {np.log10(beta)}")
        print(f"gammalog: {np.log10(gamma)}")
        print(f"n: {n}")
    
    tmp_nl_force = None
    if i % 3 == 0 or i % 3 == 1: #x or y 
        tmp_nl_force = BoucWenForce(Ls, Lf, A,
                                   beta,
                                   gamma,
                                   n)
    else:
        tmp_nl_force = GenPolyForce(Ls, Lf, np.array([[Kn]]), np.array([[1]])) #Linear penalty stiffness for the normal dimension
    
    vib_sys.add_nl_force(tmp_nl_force)


Astart = -9
Aend = -4.7

# Normal - settings for higher accuracy as used in previous papers
h_max = 3 # harmonics 0, 1, 2, 3
Nt = 1<<7 # 2**7 = 128 AFT steps 

ds = 0.08
dsmax = 0.125*1.4
dsmin = 0.02
# Adjust weighting of amplitude v. other in continuation to hopefully 
# reduce turning around. Higher puts more emphasis on continuation 
# parameter (amplitudFracLam = 0.9
###############################################################################
####### 3. Prestress Analysis                                           #######
###############################################################################
h_max = 3
h = np.array(range(h_max + 1))
Nhc = hutils.Nhc(h)

Fv = system_matrices['Fv'][:, 0]
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


eigvals, eigvecs = static_solver.eigs(Kpre, system_matrices['M'], 
                                      subset_by_index=[0, 9])

eigvals, eigvecs = hutils_sam.process_eigenpairs(eigvals, eigvecs)
print(f"First Eiegenvalue: {eigvals[0]}")
#Fix negative eigenvalues -- ask Brake about these


#eigvals = eigvals[1:]
#eigvecs = eigvecs[:, 1:] #BANDAID SOLUTION TO FIX LARGE NEG EIGENVALUE

###############################################################################
####### 12. Updated Damping Matrix After Prestress                      #######
###############################################################################

# This block resets the damping matrix after prestress analysis to 
# achieve the desired levels of viscous linear damping for the first and second
# bending modes

# First and Second Bending Modes damping ratios (taken from experiments on BRB)
desired_zeta = np.array([0.087e-2, 0.034e-2]) 

# 1st and 2nd bending mode = total 1st and 3rd modes
omega_12 = np.array([np.sqrt(eigvals)[0:3:2]]).reshape(2, 1) 

# Matrix problem for proportional damping
prop_mat = np.hstack((1/(2.0*omega_12), omega_12/2.0))

pre_ab = np.linalg.solve(prop_mat, desired_zeta)

vib_sys.set_new_C(C=pre_ab[0]*vib_sys.M + pre_ab[1]*Kpre)

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
Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] 

Uwxa0 = np.zeros(Nhc*Ndof + 3)

# Static Displacements (prediction for 0th harmonic)
Uwxa0[:Ndof] = Xpre

# Mode Shape (from linearized system for prediction)
mode_ind = 0
Uwxa0[2*Ndof:3*Ndof] = np.real(eigvecs[:, mode_ind])

# Linear Frequency (for prediction of low amplitude EPMC)
Uwxa0[-3] = np.sqrt(np.real(eigvals[mode_ind]))

# Initial Damping (low amplitude as prescribed)
zeta = desired_zeta[0] # This is what mass/stiff prop damping should give
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
plt.title("Bouc-Wen: " + np.array2string(np.round(lparams, 5)))
plt.show()

np.save('data/Uwxa_full_bouc.npy', Uwxa_full)

