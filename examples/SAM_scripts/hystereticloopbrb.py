#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:43:19 2024

@author: samuelmccaulley
"""

import numpy as np
import sys
sys.path.append('../..')
import tmdsimpy.nlutils as nlutils
from tmdsimpy.hysteretic_loop_fitness import *
import matplotlib.pyplot as plt
from tmdsimpy.nlforces.arctangent_stiffness import ArctangentStiffness


solve_data = np.load('data/iwan_epmc.npz')

Uwxa_full = solve_data['Uwxa_full']

Uwxa_nl_full = solve_data['Uwxa_nl_full']
h = solve_data['h']
Ndof = solve_data['Ndof']
Q = solve_data['Q']
iwan_parameters = solve_data['iwan_parameters']
patch_areas = solve_data['patch_areas']
Nnl = Q.shape[0]
dof = 5
Fs = iwan_parameters[0] * patch_areas[dof // 3]
Kt = iwan_parameters[1] * patch_areas[dof // 3]
Fn = iwan_parameters[-1] * patch_areas[dof // 3]
beta = iwan_parameters[3]
chi = iwan_parameters[2]
beginning_amplitude = 10**Uwxa_full[0, -1]

phi_max = Fs*(1 + beta)/(Kt * (beta + (chi + 1)/(chi + 2)))
Kt_true = Kt*(1 - (beginning_amplitude/phi_max)**(1 + chi)/(chi + 2)/(beta + 1))

print(iwan_parameters)
line = -1


U0norm = np.zeros((Uwxa_full.shape[0]))
for line in range(Uwxa_full.shape[0]):
    U0norm[line] = np.linalg.norm(Uwxa_nl_full[line, dof:-3:Ndof]*10**Uwxa_nl_full[line, -1])


b, s = ArctangentStiffness.incomplete_slip_parameters(np.log10(U0norm), Uwxa_full[:, -3], Uwxa_full[:, -2])
X0 = np.atleast_2d(Uwxa_full[line, dof:-3:Ndof]).T*(10**Uwxa_full[line, -1])
U0 = np.atleast_2d(Uwxa_nl_full[line, dof:-3:Nnl]).T*(10**Uwxa_nl_full[line, -1])
print(U0)
lam = Uwxa_full[line, -3]


time, disp, iwan_force = nlutils.hysteresis_loop(1 << 10, h, U0, lam, 'iwan', [Kt, Fs, iwan_parameters[2], iwan_parameters[3]])
plt.plot(disp, iwan_force)
plt.title("Hysteresis loop Iwan")
plt.show()

time, disp, arc_force = nlutils.hysteresis_loop(1 << 10, h, U0, lam, 'arctangent', np.array([Kt_true, b, s]))
plt.plot(disp, arc_force)
plt.title("Hysteresis loop Arctangent")
plt.show()

ks_iwan = np.zeros((Uwxa_full.shape[0]))
ks_arc = np.zeros((Uwxa_full.shape[0]))

for line in range(Uwxa_full.shape[0]):
   Uwxa = Uwxa_full[line, :]
   omega = Uwxa[-3]
   amplitude = Uwxa[-1]
   X0 = np.atleast_2d(Uwxa_full[line, dof:-3:Ndof]).T*(10**Uwxa_full[line, -1])
   U0 = np.atleast_2d(Uwxa_nl_full[line, dof:-3:Nnl]).T*(10**Uwxa_nl_full[line, -1])
   
   time, disp, iwan_force = nlutils.hysteresis_loop(1 << 10, h, U0, lam, 'iwan', [Kt, Fs, iwan_parameters[2], iwan_parameters[3]])
   ks_iwan[line] = (iwan_force[np.argmax(disp)] - iwan_force[np.argmin(disp)])/(max(disp) - min(disp))
   
   time, disp, arc_force = nlutils.hysteresis_loop(1 << 10, h, U0, lam, 'arctangent', [Kt_true, b, s])
   ks_arc[line] = (arc_force[np.argmax(disp)] - arc_force[np.argmin(disp)])/(max(disp) - min(disp))
   
   if line == 100:
        # Create figure with two subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(disp, iwan_force)
        ax1.set_title("Hysteresis loop Iwan")
        
        ax2.plot(disp, arc_force)
        ax2.set_title("Hysteresis loop Arctangent")
        
        # Adjust spacing and display
        plt.tight_layout()
        plt.show()

    
plt.plot(Uwxa_full[:, -3], ks_iwan)
plt.plot(Uwxa_full[:, -3], ks_arc)
plt.legend(('Iwan secant stiffness', 'Arctangent secant stiffness'))
plt.show()