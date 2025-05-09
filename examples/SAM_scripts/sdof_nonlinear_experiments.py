#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:21:49 2024

@author: samuelmccaulley
"""
import sys
#sys.path.append('../sdof_iwan_epmc.py')
from sdof_iwan_epmc import *
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4
from tmdsimpy import nlutils
from time import time
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, QuadraticModel, ExponentialModel, SineModel
import lmfit
from scipy.stats import linregress


def dksdw(m, k, kt):
    # kt,  N/m, Match Jenkins
    Fs = 0.2  # N, Match Jenkins
    chi = -0.5  # Have a more full hysteresis loop than chi=0.0
    beta = 0.0  # Smooth Transition
    Q = np.array([[1]])
    T = np.array([[1]])
    
    nlforce = VectorIwan4(Q, T, kt, Fs, chi, beta)
    
    config = None
    
    freq_diffs, ks_diffs, success = sdof_nonlinear_experiment(m, 0.01, k, nlforce, config)
    
    finite_difference = np.diff(ks_diffs[:-1][np.diff(freq_diffs) != 0])/np.diff(freq_diffs[:-1][np.diff(freq_diffs) != 0])
    
    hist, bin_edges = np.histogram(finite_difference, bins=5000)  # Adjust bins as needed
    
    # Find the bin with the maximum frequency
    max_bin_idx = np.argmax(hist)
    mode_value = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    
    plt.plot(finite_difference)
    plt.title(f'dkskw(omega), {mode_value}')
    plt.show()
    

    
    #print("Approximated Mode:", mode_value)
    
    
    #plt.plot(freq_diffs, ks_diffs)
    #plt.plot(np.linspace(-0.02, 0), mode_value*np.linspace(-0.02, 0))
    
    return (mode_value, success) #dksdw in linear regime


# # Generate random values
# m_vals = np.random.uniform(0.1, 100, size=200)  # Random m values
# k_vals = np.random.uniform(0.1, 100, size=200)  # Random k values
# kt_vals = np.random.uniform(0.1, 100, size=200)  # Random kt values

# # Evaluate dksdw and filter based on succ
# filtered_results = [(m, k, kt, der) for m, k, kt in zip(m_vals, k_vals, kt_vals) 
#                     if (der := dksdw(m, k, kt))[1]]  # Extract only when succ is True

# # Split filtered results into x and y components
# filtered_x = [(m, k, kt) for m, k, kt, _ in filtered_results]
# filtered_y = [der[0] for _, _, _, der in filtered_results]

# m_vals_filtered, k_vals_filtered, kt_vals_filtered = zip(*filtered_x)

m_vals = np.random.uniform(0.1, 100, size=1000)
m = 1
k_vals = np.random.uniform(0.1, 100, size=1000)
k = 1.5
kt = 0.25


#filtered_results = [(m, k, kt, der) for m, k in zip(m_vals, k_vals) 
                     #if (der := dksdw(m, k, kt))[1]]  # Extract only when succ is True

filtered_results = [(m, k, kt, der) for m, k in zip(m_vals, k_vals)
                     if (der := dksdw(m, k, kt))[1]]  # Extract only when succ is True
#%% Print Relationship
fil_der = [der[0] for _, _, _, der in filtered_results if der[0] >= 0.0002]
fil_m = [m for m, _, _, der in filtered_results if der[0] >= 0.0002]
fil_k = [k for _, k, _, der in filtered_results if der[0] >= 0.0002]


plt.plot([m**0.25 for m in fil_m], fil_der, '.')
plt.xlabel("m^1/4")
plt.ylabel("slope")
plt.show()

plt.plot([1/np.sqrt(k) for k in fil_k], fil_der, '.')
plt.xlabel('1/sqrt(k)')
plt.ylabel('slope')
plt.show()

plt.plot([m**0.25/np.sqrt(k) for k, m in zip(fil_k, fil_m)], fil_der, '.')



#%% Model Fitting
import lmfit

def model_form(m, k, a):
    return a*m**0.25/np.sqrt(k)


model = lmfit.Model(model_form, independent_vars=['m', 'k'])  # Specify independent variables

# Initial guess for the parameters
params = lmfit.Parameters()
params.add('a', value=1)  # Initial guess for 'a'


# Fit the model
result = model.fit(fil_der, params, m=fil_m, k=fil_k)

# Print fit report
print(result.fit_report())


