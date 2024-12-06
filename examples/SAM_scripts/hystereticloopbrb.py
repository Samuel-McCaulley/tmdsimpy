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
import pygad


solve_data = np.load('data/iwan_epmc.npz')

Uwxa_full = solve_data['Uwxa_full']
Uwxa_nl_full = solve_data['Uwxa_nl_full']
h = solve_data['h']
Ndof = solve_data['Ndof']
Q = solve_data['Q']
iwan_parameters = solve_data['iwan_parameters']
patch_area = solve_data['patch_area']
Nnl = Q.shape[0]
dof = 9
line = 85
X0 = np.atleast_2d(Uwxa_full[line, dof:-3:Ndof]).T*(10**Uwxa_full[line, -1])
U0 = np.atleast_2d(Uwxa_nl_full[line, dof:-3:Nnl]).T*(10**Uwxa_nl_full[line, -1])

lam = Uwxa_full[line, -3]


time, disp, force = nlutils.hysteresis_loop(1 << 10, h, U0, lam, 'iwan', iwan_parameters[:-1])
plt.plot(disp, force)