import numpy as np
import sys
import io
from scipy import io as sio
sys.path.append('../..')
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.nlforces.iwan4_element import Iwan4Force
from tmdsimpy.nlforces.iwan_patch import IwanPatch
from tmdsimpy.nlforces.general_poly_stiffness import GenPolyForce
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.jax.solvers import NonlinearSolverOMP
import matplotlib.pyplot as plt

from tmdsimpy.nlforces.bouc_wen_new import BoucWenForceNew

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.continuation import Continuation


import tmdsimpy.nlutils as nlutils


system_fname = './data/brb_iwan4_mesh.mat'
system_matrices = sio.loadmat(system_fname)

X0 = system_matrices['X0']
Q = system_matrices['Qxyn']
T = system_matrices['Txyn']

U = Q @ X0

lpsci = [1, 0, 0, 0]
lpars = [10, 1, -1, 4]
pars = nlutils.paramexp(lpars, lpsci)


Uwxa_full = np.load('data/Uwxa_full_iwan.npy')

dof = 7
line = 50
Ndof = Q.shape[1]
X0 = np.atleast_2d(Uwxa_full[line, dof:-3:Ndof]).T*(10**Uwxa_full[line, -1])
h_max = 3
h = np.array(range(h_max+1))
lam = Uwxa_full[line, -3]

parameters = [1e6,
              100000,
              100000,
              1,
              ]

time, disp, forces = nlutils.hysteresis_loop(1<<13, h, X0, lam, 'bouc-wen', parameters)

plt.plot(disp, forces)