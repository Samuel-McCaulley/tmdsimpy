import sys
import os
from scipy import io as sio
import numpy as np
from tmdsimpy.tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.tmdsimpy.jax.solvers import NonlinearSolverOMP
import tmdsimpy.tmdsimpy.utils.harmonic as hutils

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

iwan_parameters = [5.26365059447323*10**6,	16.5477290857517*10**6,	12.3118961605607,	0.848501586272668] #Replace with actual parameters

for i in range(Nnl):
    Ls = Q[i:i+1, :];
    Lf = T[:, i:i+1];
    
    tmp_nl_force = VectorIwan4(Ls, Lf, iwan_parameters[0],
                               iwan_parameters[1],
                               iwan_parameters[2],
                               iwan_parameters[3])
    
    vib_sys.add_nl_force(tmp_nl_force)
###############################################################################
####### 4. Create Forcing Vector                                        #######
###############################################################################

h_max = 3
h = np.array(range(h_max + 1))
Nhc = hutils.Nhc(h)



Fl = np.zeros(Nhc*Ndof) #Forcing vector
Fv = system_matrices['Fv'][:, 0]
prestress = 12249.0 #stolen from brb_epmc
Fl[:Ndof] = prestress*Fv #Static force

# EPMC phase constraint - No cosine component at accel
Fl[Ndof:2*Ndof] = system_matrices['R'][2, :] 


pre_fun = lambda U, calc_grad=True : vib_sys.static_res(U, Fv*prestress)

static_config={'max_steps' : 30,
                'reform_freq' : 1,
                'verbose' : True, 
                'xtol'    : None, 
                'stopping_tol' : ['xtol']
                }

# Custom Newton-Raphson solver
static_solver = NonlinearSolverOMP(config=static_config) 

X0 = np.array(system_matrices['X0'])

Xpre, R, dRdX, sol = static_solver.nsolve(pre_fun, X0,
                                          verbose=True, xtol=1e-13)

# =============================================================================
# ###############################################################################
# ####### 6. Create EMPC Initial Guess                                    #######
# ###############################################################################
# 
# X0 = np.array(system_matrices['X0'])
# Nt = 1<<7
# def epmc_fun(Uwxa):
#     R, dRdX, _ = vib_sys.epmc_res(np.hstack((Uwxa, np.array([-5.5]))), Fl, h, Nt=Nt, calc_grad = True)
#     print('done with eval')
#     print(np.linalg.norm(R))
#     return R, dRdX
# 
# solver = NonlinearSolver()
# print('here')
# X, R, dXdR, sol = solver.nsolve(epmc_fun, X0, verbose=True)
# 
# =============================================================================



