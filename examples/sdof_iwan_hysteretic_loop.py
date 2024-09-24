#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 20:52:46 2024

@author: samuelmccaulley
"""

import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy.vibration_system import VibrationSystem
from tmdsimpy.solvers import NonlinearSolver
from tmdsimpy.continuation import Continuation
import tmdsimpy.utils.harmonic as hutils
import matplotlib.pyplot as plt

def harmonic_displacement(coefficients, omega, t):
    """
    Computes the displacement x(t) based on harmonic balance coefficients.

    Args:
        coefficients (list): List of coefficients [h0, h1cosine, h1sine, ..., hncosine, hnsine].
        omega (float): Fundamental frequency.
        t (float): Time at which to evaluate the displacement.

    Returns:
        float: The displacement at time t.
    """
    h0 = coefficients[0]
    displacement = h0  # Start with the constant term
    
    # Loop over pairs of cosine and sine coefficients
    for n in range(1, len(coefficients) // 2 + 1):
        h_cosine = coefficients[2 * n - 1]  # Cosine coefficient for harmonic n
        h_sine = coefficients[2 * n]        # Sine coefficient for harmonic n
        displacement += h_cosine * np.cos(n * omega * t) + h_sine * np.sin(n * omega * t)

    return displacement

def harmonic_velocity(coefficients, omega, t):
    """
    Computes the velocity xdot(t) based on harmonic balance coefficients.

    Args:
        coefficients (list): List of coefficients [h0, h1cosine, h1sine, ..., hncosine, hnsine].
        omega (float): Fundamental frequency.
        t (float): Time at which to evaluate the velocity.

    Returns:
        float: The velocity at time t.
    """
    velocity = 0
    
    # Loop over pairs of cosine and sine coefficients
    for n in range(1, len(coefficients) // 2 + 1):
        h_cosine = coefficients[2 * n - 1]  # Cosine coefficient for harmonic n
        h_sine = coefficients[2 * n]        # Sine coefficient for harmonic n
        velocity += -n * omega * h_cosine * np.sin(n * omega * t) + n * omega * h_sine * np.cos(n * omega * t)

    return velocity

###############################################################################
####### System parameters                                               #######
###############################################################################

m = 1 # kg
c = 0.01 # kg/s
k = 0.75 # N/m


kt = 0.25 # N/m, Match Jenkins
Fs = 0.2 # N, Match Jenkins
chi = -0.5 # Have a more full hysteresis loop than chi=0.0
beta = 0.0 # Smooth Transition
Nsliders = 100

ab_damp = [c/m, 0]


h_max = 8 # Run 3 for paper or 8 for verification.
Nt = 1<<10 # number of steps for AFT evaluations

lam0 = 0.1 #Forcing Frequency? -SAM
lam1 = 1
lambdas = np.linspace(lam0, lam1, 10)




###############################################################################
####### Model Construction                                              #######
###############################################################################

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])

iwan_force = VectorIwan4(Q, T, kt, Fs, chi, beta, Nsliders=Nsliders, alphasliders=1.0)

# Setup Vibration System
M = np.array([[m]])

K = np.array([[k]])

ab_damp = [c/m, 0]

vib_sys = VibrationSystem(M, K, ab=ab_damp)
solver = NonlinearSolver()


fmag = 1 # Unity -SAM

h = np.array(range(h_max+1))
t_steps = np.linspace(0, 100, 1000)
cmap = plt.cm.get_cmap("coolwarm")

# Setup
Ndof = 1
Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)

# External Forcing Vector
Fl = np.zeros(Nhc*Ndof)
Fl[1] = 1 # Cosine Forcing at Fundamental Harmonic
vib_sys.add_nl_force(iwan_force)


plt.figure(figsize=[8, 6])
for i, lam in enumerate(lambdas):
    
    color = cmap(i / (len(lambdas) - 1))
    fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam)), fmag*Fl, h, Nt=Nt)[0:2]
    # Initial Nonlinear Solution Point
    
    U0 = np.zeros_like(Fl)
    
    X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)
    #Create time history

    x_t = np.array([harmonic_displacement(X, lam, t) for t in t_steps]).reshape(-1, 1)
    xdot_t = np.array([harmonic_velocity(X, lam, t) for t in t_steps]).reshape(-1, 1)
    
    cst = np.ones([len(t_steps), Nhc])
    
    forces = iwan_force.local_force_history(x_t, xdot_t, h, cst, X[0])
    
    plt.plot(x_t, forces[0], label = f"lam = {lam:.3f}", color = color)


plt.title(f"Iwan Hysteretic Curve")
plt.xlabel("u")
plt.ylabel("F")
plt.legend(loc = 'upper right')



