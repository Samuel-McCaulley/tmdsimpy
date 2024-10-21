#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:08:58 2024

@author: samuelmccaulley
"""


import sys
import numpy as np

sys.path.append('..')
# import tmdsimpy

from tmdsimpy.nlforces.bouc_wen import BoucWenForce
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

A = 1
beta = 1
gamma = 1
ns =  np.linspace(0.1, 10, 10)


h_max = 3 # Run 3 for paper or 8 for verification.
Nt = 1<<10 # number of steps for AFT evaluations

lam=1

# Nonlinear Force
Q = np.array([[1.0]])
T = np.array([[1.0]])


M = np.array([[m]])
K = np.array([[k]])
ab_damp = [c/m, 0]

cmap = plt.cm.get_cmap("coolwarm")


for i, n in enumerate(ns):
    bouc_wen_force = BoucWenForce(Q, T, A, beta, gamma, n)
    
    vib_sys = VibrationSystem(M, K, ab=ab_damp)
    solver = NonlinearSolver()

    fmag = 1
    h = np.array(range(h_max +1))
    t_steps = np.linspace(0, 100, 1000)

    # Setup
    Ndof = 1
    Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)

    Fl = np.zeros(Nhc*Ndof)
    Fl[1] = 1
    vib_sys.add_nl_force(bouc_wen_force)

    
    color = cmap(i / (len(ns) - 1))
    fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam)), fmag*Fl, h, Nt=Nt)[0:2]
    # Initial Nonlinear Solution Point
    
    U0 = np.zeros_like(Fl)
    
    X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)
    #Create time history

    x_t = np.array([harmonic_displacement(X, lam, t) for t in t_steps]).reshape(-1, 1)
    xdot_t = np.array([harmonic_velocity(X, lam, t) for t in t_steps]).reshape(-1, 1)
    
    cst = np.ones([len(t_steps), Nhc])
    
    forces = bouc_wen_force.local_force_history(x_t, xdot_t, h, cst, X[0])
    
    plt.plot(x_t, forces[0], label = f"n = {n:.3f}", color = color)


plt.title(f"Bouc-Wen Hysteretic Curve")
plt.xlabel("u")
plt.ylabel("F")
plt.legend(loc = 'upper right')
        
# =============================================================================
# n = 1.5
# gammas = np.linspace(0.1, 10, 10)
# 
# for i, gamma in enumerate(gammas):
#     bouc_wen_force = BoucWenForce(Q, T, A, beta, gamma, n)
#     
#     vib_sys = VibrationSystem(M, K, ab=ab_damp)
#     solver = NonlinearSolver()
# 
#     fmag = 1
#     h = np.array(range(h_max +1))
#     t_steps = np.linspace(0, 100, 1000)
# 
#     # Setup
#     Ndof = 1
#     Nhc = hutils.Nhc(h) # Number of Harmonic Components (2*h_max + 1)
# 
#     Fl = np.zeros(Nhc*Ndof)
#     Fl[1] = 1
#     vib_sys.add_nl_force(bouc_wen_force)
# 
#     
#     color = cmap(i / (len(ns) - 1))
#     fun = lambda U : vib_sys.hbm_res(np.hstack((U, lam)), fmag*Fl, h, Nt=Nt)[0:2]
#     # Initial Nonlinear Solution Point
#     
#     U0 = np.zeros_like(Fl)
#     
#     X, R, dRdX, sol = solver.nsolve(fun, fmag*U0)
#     #Create time history
# 
#     x_t = np.array([harmonic_displacement(X, lam, t) for t in t_steps]).reshape(-1, 1)
#     xdot_t = np.array([harmonic_velocity(X, lam, t) for t in t_steps]).reshape(-1, 1)
#     
#     cst = np.ones([len(t_steps), Nhc])
#     
#     forces = bouc_wen_force.local_force_history(x_t, xdot_t, h, cst, X[0])
#     
#     plt.plot(x_t, forces[0], label = f"n = {n:.3f}", color = color)
# 
# =============================================================================

# =============================================================================
# plt.title(f"Bouc-Wen Hysteretic Curve")
# plt.xlabel("u")
# plt.ylabel("F")
# plt.legend(loc = 'upper right')
# =============================================================================
        




