import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from tmdsimpy.nlforces.arctangent_stiffness import *
from tmdsimpy.nlforces.vector_iwan4 import *
from tmdsimpy.nlforces.bouc_wen import *
from tmdsimpy.utils.harmonic import *
import matplotlib.pyplot as plt
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.vibration_system import VibrationSystem
from scipy import io as sio
import time

Uwxa_full = np.load('data/Uwxa_full_iwan.npy')

Q = np.array([[1]])
T = np.array([[1]])

#Compute center-shift
max_omega = Uwxa_full[:, -3].max()
min_omega = Uwxa_full[:, -3].min()
center_omega = 1/2*(max_omega + min_omega)
omega_index = np.argmin(np.abs(Uwxa_full[:, -3] - center_omega))

kt = 1.25
m = 1
c = 0.01
k = 1.5

"""
Solve for Parameters
"""
s = Uwxa_full[omega_index, -1]
b = 4

arctangent_force = ArctangentStiffness(Q, T, kt, b, s)
h = np.array([0, 1, 2, 3])
cst = np.ones((1 << 8, hutils.Nhc(h)))

line = -1 #arbitrary
X0 = np.atleast_2d(Uwxa_full[line, :-3]).T * 10**Uwxa_full[line, -1]

Ut = hutils.time_series_deriv(1 << 8, h, X0, 0)
Utdot = hutils.time_series_deriv(1 << 8, h, X0, 1)


iwan_force = VectorIwan4(Q, T, kt, 0.2, -0.5, 0)


'''
Timings
'''
start = time.time()
for i in range(1000):
    arctangent_force_history = arctangent_force.local_force_history(Ut, Utdot, h, cst, X0[0])
print(f"It took {time.time() - start} seconds for Arctangent")

start = time.time()
for i in range(1000):
    iwan_force_history = iwan_force.local_force_history(Ut, Utdot, h, cst, X0[0])
print(f"It took {time.time() - start} seconds for Iwan")

plt.plot(Ut, arctangent_force_history[0])
plt.plot(Ut, iwan_force_history[0])
plt.show()

